import numpy as np

# Parse the current log output
data = """
/media/tam/DATA/3D/CG-photo/results_replica_pa6_v2/replica_rgbd_0
office0 0.005154 0.003238 36.71789054679871 0.9562277011573315 0.06920733383484184 37.093448357581146 366.68461490109104 0.002302
office1 0.004497 0.005932 37.886784677505496 0.9548030355274677 0.06321815969329328 39.16638788862107 332.64601518719576 0.001687
office2 None None 31.13292318725586 0.9284322467446328 0.0968673038808629 35.07490043552333 287.1631426461336 None
office3 None None 31.241652061462403 0.9208932288587094 0.09143291799537838 33.6911676037197 269.21240731490934 None
office4 None None 34.299478203773496 0.9437012666165828 0.06983073620032519 29.848382240112084 255.96767595693635 None
room0 0.005269 0.002551 29.53125590133667 0.8765870414078235 0.11934230466932058 33.222701251584105 228.73130295299396 0.001944
room1 None None 30.537371397018433 0.9006701349318027 0.09698704886250198 36.2969501540625 268.955051339503 None
room2 None None 33.830223987579345 0.9406497175097466 0.05813739442639053 28.922329344970986 238.506595190386 None
/media/tam/DATA/3D/CG-photo/results_replica_pa6_v2/replica_rgbd_1
office0 0.004204 0.003515 36.80435994625091 0.9559845049977302 0.07124891596287489 36.444400033887185 344.7698084422203 0.001640
office1 None None 37.48783565235138 0.9510995167791844 0.0657983088195324 38.658173456791765 320.63899315950925 None
office2 None None 30.053573486328126 0.9220957121551037 0.1005693590277806 34.38295814235636 268.9137869363406 None
office3 None None 32.08214371490479 0.9254794169664383 0.08998637373559176 33.76458701012439 256.48792820717534 None
office4 None None 33.764600308418274 0.9427262620925904 0.07086995236668736 30.393008266961196 243.66757135880778 None
room0 None None 28.295172456741334 0.8579206634163856 0.1313136141654104 33.01213749091948 226.53788034028125 None
room1 None None 32.00122745704651 0.9150257165431976 0.08147559434734285 36.53099828929859 272.20178801677275 None
room2 None None 33.646675579071044 0.9387513892352581 0.06011851956509054 28.974529528312924 239.98998454883207 None
/media/tam/DATA/3D/CG-photo/results_replica_pa6_v2/replica_rgbd_2
office0 0.006910 0.004023 37.38893400287628 0.9602251132130623 0.06476089532952756 35.2958405322604 354.3620804294618 0.005374
office1 None None 37.082229625701906 0.9521441849470138 0.0641606077812612 38.16713546988927 335.01868907758484 None
office2 None None 30.426323078155516 0.924230682939291 0.09850343533977866 34.18905783886956 270.2155234588894 None
office3 None None 31.260651061058045 0.9214423415362835 0.08972188115119933 33.70810937740844 246.52126923508243 None
office4 None None 34.318584815979 0.9447949654757977 0.06767716765217482 30.70588188193043 250.5765561340118 None
room0 None None 30.043366018295288 0.8799226550459862 0.11201687910221517 30.872453229638285 236.64489172390813 None
room1 None None 31.58942986679077 0.9133681867718697 0.08309713310189545 33.617262663918865 264.7452879196152 None
room2 None None 31.79049551773071 0.9301669349670411 0.06472323373844847 29.3147080661477 238.9712015562883 None
"""

# Parse data
runs = {}
current_run = None

for line in data.strip().split('\n'):
    if line.startswith('/media/tam/DATA/3D/CG-photo/results_replica_pa6_v2/'):
        current_run = line.split('/')[-1]
        runs[current_run] = []
    elif line.strip() and current_run:
        parts = line.split()
        scene = parts[0]
        metrics = {
            'scene': scene,
            'T': float(parts[1]) if parts[1] != 'None' else None,
            'R': float(parts[2]) if parts[2] != 'None' else None,
            'PSNR': float(parts[3]) if parts[3] != 'None' else None,
            'SSIM': float(parts[4]) if parts[4] != 'None' else None,
            'LPIPS': float(parts[5]) if parts[5] != 'None' else None,
            'Tracking_FPS': float(parts[6]) if parts[6] != 'None' else None,
            'Rendering_FPS': float(parts[7]) if parts[7] != 'None' else None,
            'T_std': float(parts[8]) if parts[8] != 'None' else None,
        }
        runs[current_run].append(metrics)

print("="*90)
print("REPLICA PA6 V2 METRICS SUMMARY")
print("="*90)

# Overall metrics across all available data
all_T = []
all_R = []
all_PSNR = []
all_SSIM = []
all_LPIPS = []
all_Tracking_FPS = []
all_Rendering_FPS = []
all_T_std = []

for run_name, scenes in runs.items():
    print(f"\n{run_name}:")
    print("-"*90)
    for s in scenes:
        t_str = f"{s['T']:.6f}m" if s['T'] is not None else "N/A"
        r_str = f"{s['R']:.6f}" if s['R'] is not None else "N/A"
        psnr_str = f"{s['PSNR']:.2f}dB" if s['PSNR'] is not None else "N/A"
        ssim_str = f"{s['SSIM']:.4f}" if s['SSIM'] is not None else "N/A"
        lpips_str = f"{s['LPIPS']:.4f}" if s['LPIPS'] is not None else "N/A"
        
        print(f"  {s['scene']:12s}  ATE: {t_str:12s}  R: {r_str:10s}  PSNR: {psnr_str:10s}  SSIM: {ssim_str:8s}  LPIPS: {lpips_str:8s}")
        
        if s['T'] is not None: all_T.append(s['T'])
        if s['R'] is not None: all_R.append(s['R'])
        if s['PSNR'] is not None: all_PSNR.append(s['PSNR'])
        if s['SSIM'] is not None: all_SSIM.append(s['SSIM'])
        if s['LPIPS'] is not None: all_LPIPS.append(s['LPIPS'])
        if s['Tracking_FPS'] is not None: all_Tracking_FPS.append(s['Tracking_FPS'])
        if s['Rendering_FPS'] is not None: all_Rendering_FPS.append(s['Rendering_FPS'])
        if s['T_std'] is not None: all_T_std.append(s['T_std'])

print("\n" + "="*90)
print("OVERALL AVERAGES (based on available metrics)")
print("="*90)
print(f"  ATE (Translation):       {np.mean(all_T)*1000:.2f} ± {np.std(all_T)*1000:.2f} mm  (n={len(all_T)} scenes)")
print(f"  RPE (Rotation):          {np.mean(all_R)*1000:.2f} ± {np.std(all_R)*1000:.2f} mrad  (n={len(all_R)} scenes)")
print(f"  PSNR:                    {np.mean(all_PSNR):.2f} ± {np.std(all_PSNR):.2f} dB  (n={len(all_PSNR)} scenes)")
print(f"  SSIM:                    {np.mean(all_SSIM):.4f} ± {np.std(all_SSIM):.4f}  (n={len(all_SSIM)} scenes)")
print(f"  LPIPS:                   {np.mean(all_LPIPS):.4f} ± {np.std(all_LPIPS):.4f}  (n={len(all_LPIPS)} scenes)")
print(f"  Tracking FPS:            {np.mean(all_Tracking_FPS):.2f} ± {np.std(all_Tracking_FPS):.2f}  (n={len(all_Tracking_FPS)} scenes)")
print(f"  Rendering FPS:           {np.mean(all_Rendering_FPS):.2f} ± {np.std(all_Rendering_FPS):.2f}  (n={len(all_Rendering_FPS)} scenes)")

print("\n" + "="*90)
print("NOTES:")
print("="*90)
print(f"  • Total scenes evaluated: {len(all_PSNR)} out of 24 (3 runs × 8 scenes)")
print(f"  • Scenes with ATE metrics: {len(all_T)} (office0×3, office1×1, room0×1)")
print(f"  • Missing ATE for: office1(2), office2-4(3 each), room0(2), room1-2(3 each)")
print(f"  • To get complete ATE metrics, need to run evaluation for remaining scenes")
print("="*90)
