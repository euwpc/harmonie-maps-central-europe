import requests
import xml.etree.ElementTree as ET
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib
import datetime
import os
import imageio.v2 as imageio
import pandas as pd
import warnings
import time

matplotlib.use('Agg')

# Suppress cartopy download warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")

# KNMI Data Platform API key
API_KEY = "13aaa27fe3d346ebd5c246ff6a2d82c0"

# --- Helper to parse QML color ramp ---
def parse_qml_colormap(qml_file, vmin, vmax):
    tree = ET.parse(qml_file)
    root = tree.getroot()

    values = []
    colors = []

    for item in root.findall(".//colorrampshader/item"):
        value = float(item.get('value'))
        color_hex = item.get('color').lstrip('#')

        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0

        values.append(value)
        colors.append((r, g, b, 1.0))

    values, colors = zip(*sorted(zip(values, colors)))

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(values, cmap.N, clip=True)

    return cmap, norm

# --- Step 1: Get latest HARMONIE run from KNMI ---
base_url = "https://api.dataplatform.knmi.nl/open-data/v1"
headers = {"Authorization": API_KEY}

dataset_name = "harmonie-arome-cy43-p3"
dataset_version = "1.0"

list_files_url = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files"

params = {"maxKeys": 1000, "orderBy": "created", "sorting": "desc"}
response = requests.get(list_files_url, headers=headers, params=params)
response.raise_for_status()
files = response.json()["files"]

# Sort by filename descending to get latest run
files = sorted(files, key=lambda f: f["filename"], reverse=True)

latest_run_prefix = None
for f in files:
    name = f["filename"]
    if "_00000_GB" in name and name.startswith("HA43_P3_"):
        latest_run_prefix = name.split("_00000_GB")[0]
        break

if not latest_run_prefix:
    raise ValueError("No recent HARMONIE files found")

run_time_str = datetime.datetime.strptime(latest_run_prefix[-12:], "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M UTC")

# --- Step 2: Download all GRIB files for the latest run ---
grib_paths = []
for f in files:
    filename = f["filename"]
    if filename.startswith(latest_run_prefix):
        url = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{filename}/url"
        get_url_response = requests.get(url, headers=headers)
        get_url_response.raise_for_status()
        download_url = get_url_response.json()["temporaryDownloadUrl"]

        local_path = filename
        print(f"Downloading {filename}...")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as lf:
                for chunk in r.iter_content(chunk_size=8192):
                    lf.write(chunk)
        grib_paths.append(local_path)

# --- Step 3: Open with cfgrib ---
ds = xr.open_mfdataset(grib_paths, combine='by_coords', engine='cfgrib')

# Variables (standard GRIB shortNames in KNMI Cy43)
temp_c = ds['t2m'] - 273.15
dewpoint_c = ds['d2m'] - 273.15
pressure_hpa = ds['msl'] / 100
cape = ds['cape']
windgust_ms = ds['gust']
precip1h_mm = ds['tp']  # 1h accumulation in mm

# --- Step 4: Load custom colormaps ---
temp_cmap, temp_norm = parse_qml_colormap("temperature_color_table_high.qml", vmin=-40, vmax=50)
cape_cmap, cape_norm = parse_qml_colormap("cape_color_table.qml", vmin=0, vmax=5000)
pressure_cmap, pressure_norm = parse_qml_colormap("pressure_color_table.qml", vmin=870, vmax=1070)
windgust_cmap, windgust_norm = parse_qml_colormap("wind_gust_color_table.qml", vmin=0, vmax=50)
precip_cmap, precip_norm = parse_qml_colormap("precipitation_color_table.qml", vmin=0, vmax=30)

dewpoint_cmap = temp_cmap
dewpoint_norm = Normalize(vmin=-40, vmax=50)

# --- Step 5: Helper ---
def get_analysis(var):
    if 'step' in var.dims:
        return var.isel(step=0)
    return var

# --- Step 6: Central Europe view (full KNMI coverage) ---
views = {
    'central_europe': {'extent': [-5, 20, 47, 60], 'suffix': ''}
}

variables = {
    'temperature': {'var': temp_c, 'cmap': temp_cmap, 'norm': temp_norm, 'unit': '°C', 'title': '2m Temperature (°C)',
                    'levels': [-40, -38, -36, -34, -32, -30, -28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]},
    'dewpoint':    {'var': dewpoint_c, 'cmap': dewpoint_cmap, 'norm': dewpoint_norm, 'unit': '°C', 'title': '2m Dew Point (°C)',
                    'levels': [-40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    'pressure':    {'var': pressure_hpa, 'cmap': pressure_cmap, 'norm': pressure_norm, 'unit': 'hPa', 'title': 'MSLP (hPa)',
                    'levels': [890, 900, 910, 915, 920, 925, 929, 933, 938, 942, 946, 950, 954, 958, 962, 965, 968, 972, 974, 976, 978, 980, 982, 984, 986, 988, 990, 992, 994, 996, 998, 1000, 1002, 1004, 1006, 1008, 1010, 1012, 1014, 1016, 1018, 1020, 1022, 1024, 1026, 1028, 1030, 1032, 1034, 1036, 1038, 1040, 1042, 1044, 1046, 1048, 1050, 1052, 1054, 1056, 1058, 1060, 1062, 1064]},
    'cape':        {'var': cape, 'cmap': cape_cmap, 'norm': cape_norm, 'unit': 'J/kg', 'title': 'CAPE (J/kg)',
                    'levels': [0, 20, 40, 100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2800, 3200, 3600, 4000, 4500, 5000]},
    'windgust':    {'var': windgust_ms, 'cmap': windgust_cmap, 'norm': windgust_norm, 'unit': 'm/s', 'title': 'Wind Gust (m/s)',
                    'levels': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    'precipitation': {'var': precip1h_mm, 'cmap': precip_cmap, 'norm': precip_norm, 'unit': 'mm', 'title': '1h Precipitation (mm)',
                      'levels': [0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 30]},
}

# --- Generate Central Europe maps ---
for view_key, view_conf in views.items():
    extent = view_conf['extent']
    suffix = view_conf['suffix']
    lon_min, lon_max, lat_min, lat_max = extent

    for var_key, conf in variables.items():
        data = get_analysis(conf['var'])

        # Min/max in view
        try:
            cropped_data = data.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))  # latitude descending
            min_val = float(cropped_data.min(skipna=True))
            max_val = float(cropped_data.max(skipna=True))
        except:
            min_val = float(data.min(skipna=True))
            max_val = float(data.max(skipna=True))

        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        data.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap=conf['cmap'], norm=conf['norm'], levels=100,
                           cbar_kwargs={'label': conf['unit'], 'shrink': 0.8, 'pad': 0.05})

        if var_key == 'pressure':
            cl = data.plot.contour(ax=ax, transform=ccrs.PlateCarree(), colors='black', linewidths=0.8, levels=conf['levels'])
            ax.clabel(cl, inline=True, fontsize=9, fmt="%d", inline_spacing=8, use_clabeltext=True)

        ax.coastlines(resolution='10m', linewidth=1.2)
        ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=1.2, alpha=0.9)
        ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.5)
        ax.set_extent(extent)

        plt.title(
            f"HARMONIE {conf['title']}\nModel run: {run_time_str} | Analysis\n"
            f"Min: {min_val:.1f} {conf['unit']} | Max: {max_val:.1f} {conf['unit']}",
            fontsize=14, pad=20
        )
        plt.savefig(f"{var_key}{suffix}.png", dpi=170, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()

        # Animation frames
        frame_paths = []
        time_values = ds['step'].values if 'step' in ds.coords else ds['time'].values

        fig_width = 1152 / 112
        fig_height = 880 / 112

        for i in range(len(time_values)):
            if i >= 48 and (i - 48) % 3 != 0:
                continue

            fig = plt.figure(figsize=(fig_width, fig_height), dpi=112)
            ax = plt.axes(projection=ccrs.PlateCarree())
            slice_data = conf['var'].isel(step=i) if 'step' in conf['var'].dims else conf['var'].isel(time=i)

            try:
                slice_min = float(slice_data.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min)).min(skipna=True))
                slice_max = float(slice_data.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min)).max(skipna=True))
            except:
                slice_min = float(slice_data.min(skipna=True))
                slice_max = float(slice_data.max(skipna=True))

            slice_data.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap=conf['cmap'], norm=conf['norm'], levels=100)

            if var_key == 'pressure':
                cl = slice_data.plot.contour(ax=ax, transform=ccrs.PlateCarree(), colors='black', linewidths=0.8, levels=conf['levels'])
                ax.clabel(cl, inline=True, fontsize=9, fmt="%d", inline_spacing=8, use_clabeltext=True)

            ax.coastlines(resolution='10m', linewidth=1.2)
            ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=1.2, alpha=0.9)
            ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.5)
            ax.set_extent(extent)

            valid_dt = pd.to_datetime(ds['time' if 'time' in ds.coords else 'valid_time'][i].values)
            valid_str = valid_dt.strftime("%a %d %b %H:%M EET")

            plt.title(
                f"HARMONIE {conf['title']}\nValid: {valid_str} | +{i}h from run {run_time_str}\n"
                f"Min: {slice_min:.1f} {conf['unit']} | Max: {slice_max:.1f} {conf['unit']}",
                fontsize=13, pad=15
            )

            frame_path = f"frame_{var_key}{suffix}_{i:03d}.png"
            plt.savefig(frame_path, dpi=112, facecolor='white', pad_inches=0.3)
            plt.close()
            frame_paths.append(frame_path)

        video_path = f"{var_key}{suffix}_animation.mp4"
        with imageio.get_writer(video_path, fps=2, codec='libx264',
                                pixelformat='yuv420p', quality=8,
                                macro_block_size=16) as writer:
            for fp in frame_paths:
                writer.append_data(imageio.imread(fp))

        for fp in frame_paths:
            os.remove(fp)

# Cleanup GRIB files
for p in grib_paths:
    os.remove(p)

print("Central Europe (KNMI HARMONIE) maps + MP4 animations generated")
