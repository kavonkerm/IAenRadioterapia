import builtins
import warnings
from pathlib import Path
import numpy as np
import pydicom

# 1. PARCHE DE ADVERTENCIAS PARA RT-UTILS
builtins.warnings = warnings
import rt_utils.rtstruct_builder
rt_utils.rtstruct_builder.warnings = warnings
from rt_utils import RTStructBuilder

# ==========================================
# 2. CONFIGURACIÓN DE RUTAS
# ==========================================
RUTA_SERIE_CT = Path(r"C:\Users\kavon\Desktop\Seminario\PacientesHonduras\PX-001")
RUTA_RTSTRUCT = RUTA_SERIE_CT / "RTSTRUCT_1.2.826.0.1.3680043.8.498.79134355503362458553537632963833488748.dcm"

# ==========================================
# 3. RECONCILIACIÓN GEOMÉTRICA (COORDENADA Z)
# ==========================================
print("⏳ Leyendo geometría de los cortes CT...")
archivos_ct = [f for f in RUTA_SERIE_CT.glob("*.dcm") if f.name != RUTA_RTSTRUCT.name and not f.name.startswith("RTSTRUCT_corregido")]

datos_ct = []
for archivo in archivos_ct:
    try:
        ds = pydicom.dcmread(archivo, stop_before_pixels=True)
        if getattr(ds, "Modality", None) == "CT":
            # Guardamos la posición axial Z y el UID real de cada archivo
            z_pos = float(ds.ImagePositionPatient[2])
            datos_ct.append((z_pos, ds.SOPInstanceUID))
    except Exception:
        continue

# Ordenar los cortes según su coordenada física Z
datos_ct.sort(key=lambda x: x[0])
z_coords_ct = np.array([item[0] for item in datos_ct])
uids_ct = [item[1] for item in datos_ct]

print(f"✅ Se indexaron {len(datos_ct)} cortes axiales.")
print("⏳ Vinculando polígonos del RTSTRUCT a la altura Z correspondiente...")

# Abrir el RTSTRUCT original y actualizar los punteros a los cortes reales
ds_rt = pydicom.dcmread(RUTA_RTSTRUCT)

if hasattr(ds_rt, "ROIContourSequence"):
    for roi in ds_rt.ROIContourSequence:
        if hasattr(roi, "ContourSequence"):
            for contour in roi.ContourSequence:
                if hasattr(contour, "ContourData") and hasattr(contour, "ContourImageSequence"):
                    # La coordenada Z del primer punto del polígono
                    z_poligono = float(contour.ContourData[2])
                    # Buscar el corte tomográfico más cercano en altura Z
                    idx_cercano = int(np.argmin(np.abs(z_coords_ct - z_poligono)))
                    # Asignar el UID exacto del corte correspondiente
                    contour.ContourImageSequence[0].ReferencedSOPInstanceUID = uids_ct[idx_cercano]

# Guardar una versión corregida para rt-utils
ruta_rt_corregido = RUTA_SERIE_CT / "RTSTRUCT_corregido.dcm"
ds_rt.save_as(ruta_rt_corregido)
print("✅ Archivo RTSTRUCT alineado y guardado.")

# ==========================================
# 4. CARGA CON RT-UTILS Y EXTRACCIÓN
# ==========================================
print("\n⏳ Extrayendo matrices 3D con rt-utils...")
rtstruct = RTStructBuilder.create_from(
    dicom_series_path=str(RUTA_SERIE_CT), 
    rt_struct_path=str(ruta_rt_corregido),
    warn_only=True
)

organos = rtstruct.get_roi_names()
print(f"📋 Órganos encontrados ({len(organos)}): {', '.join(organos)}")

nombre_objetivo = "PROSTATA"
if nombre_objetivo in organos:
    print(f"\n⏳ Extrayendo máscara binaria para '{nombre_objetivo}'...")
    mascara_prostata = rtstruct.get_roi_mask_by_name(nombre_objetivo)
    
    total_voxeles = int(np.sum(mascara_prostata))
    print("\n" + "="*45)
    print("RESULTADO NUMÉRICO CORREGIDO:")
    print(f"Dimensiones de la matriz 3D (Z, Y, X): {mascara_prostata.shape}")
    print(f"Tipo de dato:                         {mascara_prostata.dtype}")
    print(f"Total de vóxeles de la próstata:      {total_voxeles}")
    print("="*45)
    
    if total_voxeles > 0:
        print("🎉 ¡Éxito! Los contornos se han rasterizado correctamente en la matriz 3D.")
else:
    print(f"⚠️ La estructura '{nombre_objetivo}' no se encuentra en el estudio.")