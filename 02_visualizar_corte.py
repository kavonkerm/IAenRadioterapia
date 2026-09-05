import builtins
import re
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pydicom

builtins.warnings = warnings
import rt_utils.rtstruct_builder
rt_utils.rtstruct_builder.warnings = warnings
from rt_utils import RTStructBuilder

# ==========================================
# SELECCIONA EL PACIENTE QUE DESEAS INSPECCIONAR
# ==========================================
PACIENTE_A_REVISAR = "PX-020"

RUTA_SERIE_CT = Path(r"C:\Users\kavon\Desktop\Seminario\PacientesHonduras") / PACIENTE_A_REVISAR
RUTA_RT_CORREGIDO = RUTA_SERIE_CT / "RTSTRUCT_corregido.dcm"


def obtener_nombre_prostata(organos_disponibles):
    for org in organos_disponibles:
        org_clean = org.upper().strip()
        if any(term in org_clean for term in ["LECHO", "BED"]):
            continue
        if re.search(r"\b(PROSTATA|PRÓSTATA|PROSTATE)\b", org_clean):
            return org
    for org in organos_disponibles:
        org_clean = org.upper().strip()
        if "CTV" in org_clean and ("PROST" in org_clean or "PRÓST" in org_clean):
            return org
    for org in organos_disponibles:
        org_clean = org.upper().strip()
        if re.search(r"\bGTV.*(66GY|70GY|72GY|74GY|78GY|33F|PROST)", org_clean):
            return org
    for org in organos_disponibles:
        if org.upper().strip().startswith("GTV"):
            return org
    return None


print(f"⏳ Inspeccionando: {PACIENTE_A_REVISAR}...")
rtstruct = RTStructBuilder.create_from(
    dicom_series_path=str(RUTA_SERIE_CT),
    rt_struct_path=str(RUTA_RT_CORREGIDO),
    warn_only=True
)

nombre_roi = obtener_nombre_prostata(rtstruct.get_roi_names())
print(f"🎯 Estructura seleccionada: '{nombre_roi}'")

mascara = rtstruct.get_roi_mask_by_name(nombre_roi)

# Cargar y ordenar cortes axiales
cortes_validos = []
for f in RUTA_SERIE_CT.glob("*.dcm"):
    if f.name.startswith("RTSTRUCT"):
        continue
    try:
        ds = pydicom.dcmread(f, force=True)
        if getattr(ds, "Modality", None) == "CT" and hasattr(ds, "ImagePositionPatient"):
            cortes_validos.append((float(ds.ImagePositionPatient[2]), ds))
    except Exception:
        continue

cortes_validos.sort(key=lambda x: x[0])

volumen_hu = np.zeros((512, 512, len(cortes_validos)), dtype=np.float32)
for i, (_, ds) in enumerate(cortes_validos):
    px = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", -1024.0))
    volumen_hu[:, :, i] = px * slope + intercept

# Ventana anatómica de pelvis
HU_MIN, HU_MAX = -100.0, 250.0
volumen_norm = np.clip(volumen_hu, HU_MIN, HU_MAX)

# Corte con mayor sección
area_corte = mascara.sum(axis=(0, 1))
corte_max = int(np.argmax(area_corte))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(volumen_norm[:, :, corte_max], cmap="gray", vmin=HU_MIN, vmax=HU_MAX)
ax[0].set_title(f"{PACIENTE_A_REVISAR} - TAC Corte Z = {corte_max}")
ax[0].axis("off")

ax[1].imshow(volumen_norm[:, :, corte_max], cmap="gray", vmin=HU_MIN, vmax=HU_MAX)
ax[1].imshow(
    np.ma.masked_where(~mascara[:, :, corte_max], mascara[:, :, corte_max]),
    cmap="autumn",
    alpha=0.6
)
ax[1].set_title(f"Contorno Clínico: '{nombre_roi}'")
ax[1].axis("off")

plt.tight_layout()
plt.show()