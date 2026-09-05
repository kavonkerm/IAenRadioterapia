import builtins
import concurrent.futures
import os
from pathlib import Path
import re
import time
import warnings
import numpy as np
import pydicom

# 1. PARCHE DE COMPATIBILIDAD PARA RT-UTILS
builtins.warnings = warnings
import rt_utils.rtstruct_builder
rt_utils.rtstruct_builder.warnings = warnings
from rt_utils import RTStructBuilder

# ==========================================
# 2. CONFIGURACIÓN GLOBAL
# ==========================================
RUTA_COHORTE = Path(r"C:\Users\kavon\Desktop\Seminario\PacientesHonduras")
CARPETA_SALIDA = Path(r"C:\Users\kavon\Desktop\Seminario\dataset_preprocesado")
CARPETA_SALIDA.mkdir(exist_ok=True)

HU_MIN, HU_MAX = -100.0, 250.0
PATCH_SHAPE = (160, 160, 48)  # (Y, X, Z)[cite: 2]

# Número de núcleos paralelos a usar (deja 1 libre para que la PC no se congele)
N_WORKERS = max(1, os.cpu_count() - 4)


def obtener_nombre_prostata(organos_disponibles):
    """Búsqueda jerárquica: Anatómica -> CTV -> GTV / Dosis dosimétrica."""
    # 1. Anatómico puro (excluye lechos quirúrgicos)
    for org in organos_disponibles:
        org_clean = org.upper().strip()
        if any(t in org_clean for t in ["LECHO", "BED"]):
            continue
        if re.search(r"\b(PROSTATA|PRÓSTATA|PROSTATE)\b", org_clean):
            return org, "ANATOMICO"

    # 2. CTV de próstata
    for org in organos_disponibles:
        org_clean = org.upper().strip()
        if any(t in org_clean for t in ["LECHO", "BED"]):
            continue
        if "CTV" in org_clean and ("PROST" in org_clean or "PRÓST" in org_clean):
            return org, "CTV_DOSIMETRICO"

    # 3. GTV dosimétrico / dosis absorbida / Boost prostático (ej. BOOST26Gy, GTV 66Gy)
    for org in organos_disponibles:
        org_clean = org.upper().strip()
        if re.search(r"\b(GTV|CTV|BOOST).*(66GY|70GY|72GY|74GY|78GY|26GY|33F|PROST|GY)", org_clean):
            return org, "DOSIMETRICO_BOOST"
            
    # 4. GTV dominante
    for org in organos_disponibles:
        org_clean = org.upper().strip()
        if org_clean.startswith("GTV") and not any(t in org_clean for t in ["NOD", "LINF"]):
            return org, "GTV_GENERAL"

    # 5. Casos de solo lecho postoperatorio (PX-024, PX-029)
    for org in organos_disponibles:
        org_clean = org.upper().strip()
        if "LECHO" in org_clean and "PROST" in org_clean:
            return org, "LECHO_POST_OP"

    return None, None


def extraer_parche_centrado(volumen_norm, mascara, patch_shape=PATCH_SHAPE):
    """Extrae el parche 3D centrado en el órgano aplicando padding si rebasa los límites[cite: 2]."""
    coords = np.argwhere(mascara)
    if len(coords) == 0:
        raise ValueError("La máscara binaria no contiene vóxeles activos.")

    cy, cx, cz = coords.mean(axis=0).astype(int)
    py, px, pz = patch_shape

    y1, y2 = cy - py // 2, cy + py // 2
    x1, x2 = cx - px // 2, cx + px // 2
    z1, z2 = cz - pz // 2, cz + pz // 2

    patch_img = np.full(patch_shape, -1.0, dtype=np.float32)
    patch_mask = np.zeros(patch_shape, dtype=bool)

    img_y, img_x, img_z = volumen_norm.shape
    src_y1, src_y2 = max(0, y1), min(img_y, y2)
    src_x1, src_x2 = max(0, x1), min(img_x, x2)
    src_z1, src_z2 = max(0, z1), min(img_z, z2)

    dst_y1, dst_y2 = src_y1 - y1, (src_y1 - y1) + (src_y2 - src_y1)
    dst_x1, dst_x2 = src_x1 - x1, (src_x1 - x1) + (src_x2 - src_x1)
    dst_z1, dst_z2 = src_z1 - z1, (src_z1 - z1) + (src_z2 - src_z1)

    patch_img[dst_y1:dst_y2, dst_x1:dst_x2, dst_z1:dst_z2] = volumen_norm[src_y1:src_y2, src_x1:src_x2, src_z1:src_z2]
    patch_mask[dst_y1:dst_y2, dst_x1:dst_x2, dst_z1:dst_z2] = mascara[src_y1:src_y2, src_x1:src_x2, src_z1:src_z2]

    return patch_img, patch_mask


def procesar_un_paciente(carpeta):
    """Procesa de punta a punta un paciente de forma aislada para ejecución paralela."""
    id_px = carpeta.name
    try:
        # 1. Localizar el archivo RTSTRUCT
        archivos_rt = [f for f in carpeta.glob("*.dcm") if "RTSTRUCT" in f.name and "corregido" not in f.name]
        if not archivos_rt:
            for f in carpeta.glob("*.dcm"):
                try:
                    ds_check = pydicom.dcmread(f, stop_before_pixels=True, force=True)
                    if getattr(ds_check, "Modality", None) == "RTSTRUCT":
                        archivos_rt.append(f)
                        break
                except Exception:
                    continue

        if not archivos_rt:
            return {"paciente": id_px, "estado": "ERROR: Sin RTSTRUCT", "tipo": "N/A", "voxeles": 0}

        rt_original = archivos_rt[0]

        # 2. Lectura en un solo paso: metadatos espaciales y pixel_array a la vez
        cortes_ct = []
        for f in carpeta.glob("*.dcm"):
            if f.name.startswith("RTSTRUCT"):
                continue
            try:
                ds = pydicom.dcmread(f, force=True)
                if getattr(ds, "Modality", None) == "CT" and hasattr(ds, "ImagePositionPatient"):
                    z_pos = float(ds.ImagePositionPatient[2])
                    cortes_ct.append((z_pos, ds.SOPInstanceUID, ds))
            except Exception:
                continue

        if not cortes_ct:
            return {"paciente": id_px, "estado": "ERROR: Sin cortes CT", "tipo": "N/A", "voxeles": 0}

        cortes_ct.sort(key=lambda x: x[0])
        z_coords = np.array([item[0] for item in cortes_ct])
        uids = [item[1] for item in cortes_ct]

        # 3. Corrección geométrica en memoria y guardado de RTSTRUCT sincronizado
        ds_rt = pydicom.dcmread(rt_original, force=True)
        if hasattr(ds_rt, "ROIContourSequence"):
            for roi in ds_rt.ROIContourSequence:
                if hasattr(roi, "ContourSequence"):
                    for contour in roi.ContourSequence:
                        if hasattr(contour, "ContourData") and hasattr(contour, "ContourImageSequence"):
                            z_poly = float(contour.ContourData[2])
                            idx_match = int(np.argmin(np.abs(z_coords - z_poly)))
                            contour.ContourImageSequence[0].ReferencedSOPInstanceUID = uids[idx_match]

        rt_corregido = carpeta / "RTSTRUCT_corregido.dcm"
        ds_rt.save_as(rt_corregido)

        # 4. Extracción de máscara con rt-utils
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(carpeta),
            rt_struct_path=str(rt_corregido),
            warn_only=True
        )

        nombre_prostata, tipo_roi = obtener_nombre_prostata(rtstruct.get_roi_names())
        if not nombre_prostata:
            return {"paciente": id_px, "estado": "ERROR: Sin ROI próstata", "tipo": "N/A", "voxeles": 0}

        # Descartar lechos posquirúrgicos del dataset de próstata intacta
        if tipo_roi == "LECHO_POST_OP":
            return {"paciente": id_px, "estado": "EXCLUIDO: Lecho Post-Op", "tipo": tipo_roi, "voxeles": 0}

        mascara = rtstruct.get_roi_mask_by_name(nombre_prostata)
        total_voxeles = int(np.sum(mascara))
        if total_voxeles == 0:
            return {"paciente": id_px, "estado": "ERROR: 0 vóxeles", "tipo": tipo_roi, "voxeles": 0}

        # 5. Conversión tomográfica a HU
        volumen_hu = np.zeros((512, 512, len(cortes_ct)), dtype=np.float32)
        for i, (_, _, ds) in enumerate(cortes_ct):
            px = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
            volumen_hu[:, :, i] = px * slope + intercept

        vol_clip = np.clip(volumen_hu, HU_MIN, HU_MAX)
        vol_norm = 2.0 * (vol_clip - HU_MIN) / (HU_MAX - HU_MIN) - 1.0

        # 6. Recorte y guardado comprimido
        patch_ct, patch_mask = extraer_parche_centrado(vol_norm, mascara, PATCH_SHAPE)

        np.savez_compressed(
            CARPETA_SALIDA / f"{id_px}_preprocesado.npz",
            imagen=patch_ct,
            mascara=patch_mask,
            id_paciente=id_px,
            tipo_roi=tipo_roi,
            roi_original=nombre_prostata
        )

        return {
            "paciente": id_px,
            "estado": "OK",
            "tipo": tipo_roi,
            "cortes": len(cortes_ct),
            "voxeles": total_voxeles,
            "roi": nombre_prostata
        }

    except Exception as e:
        return {"paciente": id_px, "estado": f"ERROR: {str(e)[:25]}", "tipo": "N/A", "voxeles": 0}


# ==========================================
# 3. EJECUTOR MULTIPROCESO EN PARALELO
# ==========================================
if __name__ == "__main__":
    carpetas_pacientes = sorted([d for d in RUTA_COHORTE.iterdir() if d.is_dir()])
    total_px = len(carpetas_pacientes)
    
    print(f"🚀 Procesando {total_px} pacientes en PARALELO usando {N_WORKERS} núcleos de CPU...")
    tiempo_inicio = time.time()

    resultados = []
    # Lanzar trabajadores concurrentes
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futuros = {executor.submit(procesar_un_paciente, c): c.name for c in carpetas_pacientes}
        
        for idx, futuro in enumerate(concurrent.futures.as_completed(futuros), 1):
            px_nombre = futuros[futuro]
            res = futuro.result()
            resultados.append(res)
            estado = res["estado"]
            vox = res.get("voxeles", 0)
            print(f"[{idx:02d}/{total_px}] Finalizado: {px_nombre} -> {estado} ({vox} vóxeles)")

    tiempo_total = time.time() - tiempo_inicio

    # Ordenar reporte por nombre de paciente
    resultados.sort(key=lambda x: x["paciente"])

    # ==========================================
    # 4. REPORTE FINAL
    # ==========================================
    print("\n" + "="*85)
    print(f"📊 RESUMEN DE PROCESAMIENTO PARALELO (Tiempo: {tiempo_total:.1f} seg)")
    print("="*85)
    correctos = sum(1 for r in resultados if r["estado"] == "OK")
    print(f"Pacientes válidos para entrenamiento: {correctos}/{total_px}\n")

    print(f"{'Paciente':<10} | {'Estado':<16} | {'Tipo ROI':<16} | {'Vóxeles':<10} | {'ROI Usado'}")
    print("-" * 85)
    for r in resultados:
        print(f"{r['paciente']:<10} | {r['estado']:<16} | {r.get('tipo','N/A'):<16} | {r.get('voxeles',0):<10} | {r.get('roi','N/A')}")
    print("="*85)