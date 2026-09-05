import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset, DataLoader

# Restringir PyTorch a 4 núcleos de CPU para cálculo matricial
torch.set_num_threads(4)

# pyrefly: ignore [missing-import]
from modelo_unet3d import UNet3D
# pyrefly: ignore [missing-import]
from losses import HybridDiceFocalLoss


class DatasetProstata3D(Dataset):
    """Carga los parches volumétricos de 160x160x48 vóxeles."""
    def __init__(self, lista_rutas):
        self.rutas = lista_rutas

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, idx):
        archivo = np.load(self.rutas[idx])
        
        # De (Y, X, Z) -> Transponer a (Z, Y, X) -> (48, 160, 160)
        img = np.transpose(archivo["imagen"].astype(np.float32), (2, 0, 1))
        mask = np.transpose(archivo["mascara"].astype(np.float32), (2, 0, 1))

        # Dimensión de canal único: (1, 48, 160, 160)
        tensor_img = torch.from_numpy(img).unsqueeze(0)
        tensor_mask = torch.from_numpy(mask).unsqueeze(0)

        return tensor_img, tensor_mask


def metric_dice(logits, targets, smooth=1e-5):
    """Calcula el Coeficiente de Similitud Dice (DSC) en validación."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    interseccion = (preds * targets).sum()
    return (2.0 * interseccion + smooth) / (preds.sum() + targets.sum() + smooth)


def main():
    DIR_DATASET = Path(r"C:\Users\kavon\Desktop\Seminario\dataset_preprocesado")
    DIR_SALIDA_MODELOS = Path(r"C:\Users\kavon\Desktop\Seminario\modelos_guardados")
    DIR_SALIDA_MODELOS.mkdir(exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 35
    BATCH_SIZE = 2
    LR_INICIAL = 1e-3

    # Cargar los 29 pacientes procesados
    todos_los_archivos = sorted(list(DIR_DATASET.glob("*.npz")))
    archivos_validos = []

    for f in todos_los_archivos:
        data = np.load(f)
        tipo = str(data["tipo_roi"]) if "tipo_roi" in data else "ANATOMICO"
        if tipo != "LECHO_POST_OP":
            archivos_validos.append(f)

    total_pacientes = len(archivos_validos)
    print("=" * 70)
    print(f"🚀 PIPELINE DE ENTRENAMIENTO 3D U-NET (5-FOLD CV)")
    print(f" Cohorte activa: {total_pacientes} pacientes")
    print(f" Dispositivo:    {DEVICE} (Hilos CPU activos: {torch.get_num_threads()})")
    print("=" * 70)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    resultados_folds = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(archivos_validos), 1):
        print(f"\n📁 INICIANDO FOLD {fold}/5 | Train: {len(train_idx)} px | Val: {len(val_idx)} px")
        print("-" * 55)

        train_files = [archivos_validos[i] for i in train_idx]
        val_files = [archivos_validos[i] for i in val_idx]

        train_dataset = DatasetProstata3D(train_files)
        val_dataset = DatasetProstata3D(val_files)

        # DataLoaders configurados con 4 núcleos para lectura en segundo plano
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True if DEVICE.type == "cuda" else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=2
        )

        modelo = UNet3D(in_channels=1, out_channels=1, base_filters=16).to(DEVICE)
        criterio = HybridDiceFocalLoss().to(DEVICE)
        optimizador = torch.optim.AdamW(modelo.parameters(), lr=LR_INICIAL, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode="max", factor=0.5, patience=4)

        mejor_val_dice = 0.0

        for epoch in range(1, EPOCHS + 1):
            # Fase de Entrenamiento
            modelo.train()
            train_loss = 0.0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                optimizador.zero_grad()
                logits = modelo(imgs)
                loss = criterio(logits, masks)
                loss.backward()
                optimizador.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Fase de Validación
            modelo.eval()
            val_dice = 0.0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                    logits = modelo(imgs)
                    val_dice += metric_dice(logits, masks).item()

            val_dice /= len(val_loader)
            scheduler.step(val_dice)

            if epoch % 5 == 0 or epoch == EPOCHS:
                lr_actual = optimizador.param_groups[0]["lr"]
                print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val DSC: {val_dice:.4f} | LR: {lr_actual:.1e}")

            if val_dice > mejor_val_dice:
                mejor_val_dice = val_dice
                torch.save(modelo.state_dict(), DIR_SALIDA_MODELOS / f"unet3d_fold{fold}_best.pth")

        resultados_folds.append(mejor_val_dice)
        print(f"🏆 Fold {fold} Finalizado -> Mejor Dice en Validación: {mejor_val_dice:.4f}")

    print("\n" + "=" * 70)
    print("📊 REPORTE GLOBAL DE VALIDACIÓN CRUZADA")
    print("=" * 70)
    for i, d in enumerate(resultados_folds, 1):
        print(f"Fold {i}: Dice = {d:.4f}")
    print("-" * 70)
    print(f"Dice Promedio Global (DSC): {np.mean(resultados_folds):.4f} ± {np.std(resultados_folds):.4f}")
    print(f"Pesos exportados en: {DIR_SALIDA_MODELOS.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()