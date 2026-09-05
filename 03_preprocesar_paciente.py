import builtins
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pydicom

# Parche para la librería rt-utils
builtins.warnings = warnings
import rt_utils.rtstruct_builder
rt_utils.rtstruct_builder.warnings = warnings
from rt_utils import RTStructBuilder

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
RUTA_SERIE_CT = Path(r"C:\Users\kavon\Desktop\Seminario\PacientesHonduras\PX-001")
RUTA_RT_CORREGIDO = RUTA_SERIE_CT / "RTSTRUCT_corregido.dcm"
CARPETA_SALIDA = Path(r"C:\Users\kavon\Desktop\Seminario\dataset_preprocesado")
CARPETA_SALIDA.mkdir(exist_ok=True)

# ==========================================
# 2. CARGA DE MÁSCARA BINARIA
# ==========================================
print("⏳ Cargando estructura con rt-utils...")
rtstruct = RTStructBuilder.create_from(
    dicom_series_path=str(RUTA_SERIE_CT),
    rt_struct_path=str(RUTA_RT_CORREGIDO),
    warn_only=True
)
mascara_prostata = rtstruct.get_roi_mask_by_name("PROSTATA")

# ==========================================
# 3. LECTURA Y CONVERSIÓN A HU
# ==========================================
print("⏳ Ordenando cortes y convirtiendo intensidades a Unidades Hounsfield (HU)...")
cortes_validos = []
for f in RUTA_SERIE_CT.glob("*.dcm"):
    if f.name.startswith("RTSTRUCT"):
        continue
    try:
        ds = pydicom.dcmread(f, force=True)
        if getattr(ds, "Modality", None) == "CT" and hasattr(ds, "ImagePositionPatient"):
            z_pos = float(ds.ImagePositionPatient[2])
            cortes_validos.append((z_pos, ds))
    except Exception:
        continue

cortes_validos.sort(key=lambda x: x[0])

# Reconstruir volumen 3D aplicando Rescale Slope e Intercept
volumen_hu = np.zeros((512, 512, len(cortes_validos)), dtype=np.float32)
for idx, (_, ds) in enumerate(cortes_validos):
    pixel_array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
    volumen_hu[:, :, idx] = pixel_array * slope + intercept

# ==========================================
# 4. WINDOWING Y NORMALIZACIÓN [-1, 1]
# ==========================================
# Ventana pélvica para tejidos blandos: [-100, 250] HU
HU_MIN, HU_MAX = -100.0, 250.0
volumen_clipped = np.clip(volumen_hu, HU_MIN, HU_MAX)
volumen_norm = 2.0 * (volumen_clipped - HU_MIN) / (HU_MAX - HU_MIN) - 1.0

# ==========================================
# 5. RECORTE CENTRADO EN LA PRÓSTATA (PATCH 160x160x48)
# ==========================================
coords = np.argwhere(mascara_prostata)
centro_y, centro_x, centro_z = coords.mean(axis=0).astype(int)

# Definir márgenes del parche (alto: 160, ancho: 160, profundidad: 48)
dy, dx, dz = 80, 80, 24

y1, y2 = max(0, centro_y - dy), min(volumen_norm.shape[0], centro_y + dy)
x1, x2 = max(0, centro_x - dx), min(volumen_norm.shape[1], centro_x + dx)
z1, z2 = max(0, centro_z - dz), min(volumen_norm.shape[2], centro_z + dz)

patch_ct = volumen_norm[y1:y2, x1:x2, z1:z2]
patch_mask = mascara_prostata[y1:y2, x1:x2, z1:z2]

# Guardar parche para entrenamiento en PyTorch
np.savez_compressed(
    CARPETA_SALIDA / "PX-001_preprocesado.npz",
    imagen=patch_ct,
    mascara=patch_mask
)
print(f"✅ Archivo de entrenamiento guardado en: {CARPETA_SALIDA / 'PX-001_preprocesado.npz'}")
print(f"Dimensiones del parche: {patch_ct.shape}")

# ==========================================
# 6. VISUALIZACIÓN CON CONTRASTE ANATÓMICO REAL
# ==========================================
area_por_corte = mascara_prostata.sum(axis=(0, 1))
corte_optimo = int(np.argmax(area_por_corte))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(volumen_norm[:, :, corte_optimo], cmap="gray", vmin=-1.0, vmax=1.0)
ax[0].set_title(f"Tomografía con Contraste HU Real (Corte Z = {corte_optimo})")
ax[0].axis("off")

ax[1].imshow(volumen_norm[:, :, corte_optimo], cmap="gray", vmin=-1.0, vmax=1.0)
ax[1].imshow(
    np.ma.masked_where(~mascara_prostata[:, :, corte_optimo], mascara_prostata[:, :, corte_optimo]),
    cmap="autumn",
    alpha=0.6
)
ax[1].set_title("Ground Truth de Próstata Alineado")
ax[1].axis("off")

plt.tight_layout()
plt.show()