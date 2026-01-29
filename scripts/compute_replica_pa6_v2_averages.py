import numpy as np

# Parse log.txt data
data = """
/media/tam/DATA/3D/CG-photo/results_replica_pa6_v2/replica_rgbd_0
office0 0.005154 0.003238 36.71789054679871 0.9562277011573315 0.06920733383484184 37.093448357581146 366.68461490109104 0.002302
office1 None None 37.28800719356537 0.9503916488587856 0.06673105901759117 36.938247929963104 332.64601518719576 None
office2 None None 31.13292318725586 0.9284322467446328 0.0968673038808629 35.07490043552333 287.1631426461336 None
office3 None None 31.241652061462403 0.9208932288587094 0.09143291799537838 33.6911676037197 269.21240731490934 None
office4 None None None None None None 255.96767595693635 None
room0 None None None None None None 228.73130295299396 None
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

# Parse the data
runs = {}
current_run = None

for line in data.strip().split('\n'):
    if line.startswith('/media/tam/DATA/3D/CG-photo/results_replica_pa6_v2/'):
        current_run = line.split('/')[-1]
        runs[current_run] = []
    elif line.strip() and current_run:
        parts = line.split()
        scene = parts[0]
        metrics = {}
        metrics['scene'] = scene
        metrics['T'] = float(parts[1]) if parts[1] != 'None' else None
        metrics['R'] = float(parts[2]) if parts[2] != 'None' else None
        metrics['PSNR'] = float(parts[3]) if parts[3] != 'None' else None
        metrics['SSIM'] = float(parts[4]) if parts[4] != 'None' else None
        metrics['LPIPS'] = float(parts[5]) if parts[5] != 'None' else None
        metrics['Tracking_FPS'] = float(parts[6]) if parts[6] != 'None' else None
        metrics['Rendering_FPS'] = float(parts[7]) if parts[7] != 'None' else None
        metrics['T_std'] = float(parts[8]) if parts[8] != 'None' else None
        runs[current_run].append(metrics)

# Compute averages per run
print("="*80)
print("RESULTS PER RUN")
print("="*80)

run_averages = {}
for run_name, scenes in runs.items():
    print(f"\n{run_name}:")
    print("-"*80)
    
    # Collect all values
    T_vals = [s['T'] for s in scenes if s['T'] is not None]
    R_vals = [s['R'] for s in scenes if s['R'] is not None]
    PSNR_vals = [s['PSNR'] for s in scenes if s['PSNR'] is not None]
    SSIM_vals = [s['SSIM'] for s in scenes if s['SSIM'] is not None]
    LPIPS_vals = [s['LPIPS'] for s in scenes if s['LPIPS'] is not None]
    Tracking_FPS_vals = [s['Tracking_FPS'] for s in scenes if s['Tracking_FPS'] is not None]
    Rendering_FPS_vals = [s['Rendering_FPS'] for s in scenes if s['Rendering_FPS'] is not None]
    T_std_vals = [s['T_std'] for s in scenes if s['T_std'] is not None]
    
    # Compute averages
    avg_metrics = {
        'T': np.mean(T_vals) if T_vals else None,
        'R': np.mean(R_vals) if R_vals else None,
        'PSNR': np.mean(PSNR_vals) if PSNR_vals else None,
        'SSIM': np.mean(SSIM_vals) if SSIM_vals else None,
        'LPIPS': np.mean(LPIPS_vals) if LPIPS_vals else None,
        'Tracking_FPS': np.mean(Tracking_FPS_vals) if Tracking_FPS_vals else None,
        'Rendering_FPS': np.mean(Rendering_FPS_vals) if Rendering_FPS_vals else None,
        'T_std': np.mean(T_std_vals) if T_std_vals else None,
    }
    
    run_averages[run_name] = avg_metrics
    
    print(f"  Average T (ATE):         {avg_metrics['T']:.6f} m" if avg_metrics['T'] else "  Average T (ATE):         N/A")
    print(f"  Average R:               {avg_metrics['R']:.6f}" if avg_metrics['R'] else "  Average R:               N/A")
    print(f"  Average PSNR:            {avg_metrics['PSNR']:.2f} dB" if avg_metrics['PSNR'] else "  Average PSNR:            N/A")
    print(f"  Average SSIM:            {avg_metrics['SSIM']:.4f}" if avg_metrics['SSIM'] else "  Average SSIM:            N/A")
    print(f"  Average LPIPS:           {avg_metrics['LPIPS']:.4f}" if avg_metrics['LPIPS'] else "  Average LPIPS:           N/A")
    print(f"  Average Tracking FPS:    {avg_metrics['Tracking_FPS']:.2f}" if avg_metrics['Tracking_FPS'] else "  Average Tracking FPS:    N/A")
    print(f"  Average Rendering FPS:   {avg_metrics['Rendering_FPS']:.2f}" if avg_metrics['Rendering_FPS'] else "  Average Rendering FPS:   N/A")
    print(f"  Average T_std:           {avg_metrics['T_std']:.6f} m" if avg_metrics['T_std'] else "  Average T_std:           N/A")

# Overall average across all runs
print("\n" + "="*80)
print("OVERALL AVERAGE ACROSS ALL RUNS")
print("="*80)

all_T = [v['T'] for v in run_averages.values() if v['T'] is not None]
all_R = [v['R'] for v in run_averages.values() if v['R'] is not None]
all_PSNR = [v['PSNR'] for v in run_averages.values() if v['PSNR'] is not None]
all_SSIM = [v['SSIM'] for v in run_averages.values() if v['SSIM'] is not None]
all_LPIPS = [v['LPIPS'] for v in run_averages.values() if v['LPIPS'] is not None]
all_Tracking_FPS = [v['Tracking_FPS'] for v in run_averages.values() if v['Tracking_FPS'] is not None]
all_Rendering_FPS = [v['Rendering_FPS'] for v in run_averages.values() if v['Rendering_FPS'] is not None]
all_T_std = [v['T_std'] for v in run_averages.values() if v['T_std'] is not None]

print(f"  Overall Average T (ATE):         {np.mean(all_T):.6f} m ({len(all_T)} runs)" if all_T else "  Overall Average T (ATE):         N/A")
print(f"  Overall Average R:               {np.mean(all_R):.6f} ({len(all_R)} runs)" if all_R else "  Overall Average R:               N/A")
print(f"  Overall Average PSNR:            {np.mean(all_PSNR):.2f} dB ({len(all_PSNR)} runs)" if all_PSNR else "  Overall Average PSNR:            N/A")
print(f"  Overall Average SSIM:            {np.mean(all_SSIM):.4f} ({len(all_SSIM)} runs)" if all_SSIM else "  Overall Average SSIM:            N/A")
print(f"  Overall Average LPIPS:           {np.mean(all_LPIPS):.4f} ({len(all_LPIPS)} runs)" if all_LPIPS else "  Overall Average LPIPS:           N/A")
print(f"  Overall Average Tracking FPS:    {np.mean(all_Tracking_FPS):.2f} ({len(all_Tracking_FPS)} runs)" if all_Tracking_FPS else "  Overall Average Tracking FPS:    N/A")
print(f"  Overall Average Rendering FPS:   {np.mean(all_Rendering_FPS):.2f} ({len(all_Rendering_FPS)} runs)" if all_Rendering_FPS else "  Overall Average Rendering FPS:   N/A")
print(f"  Overall Average T_std:           {np.mean(all_T_std):.6f} m ({len(all_T_std)} runs)" if all_T_std else "  Overall Average T_std:           N/A")

# Per-scene average across all runs
print("\n" + "="*80)
print("AVERAGE PER SCENE (across all 3 runs)")
print("="*80)

all_scenes = set()
for run_scenes in runs.values():
    for s in run_scenes:
        all_scenes.add(s['scene'])

scene_data = {}
for scene_name in sorted(all_scenes):
    scene_data[scene_name] = {
        'T': [], 'R': [], 'PSNR': [], 'SSIM': [], 'LPIPS': [],
        'Tracking_FPS': [], 'Rendering_FPS': [], 'T_std': []
    }
    
    for run_scenes in runs.values():
        for s in run_scenes:
            if s['scene'] == scene_name:
                for key in scene_data[scene_name].keys():
                    if s[key] is not None:
                        scene_data[scene_name][key].append(s[key])

for scene_name in sorted(all_scenes):
    print(f"\n{scene_name}:")
    data = scene_data[scene_name]
    print(f"  T (ATE):       {np.mean(data['T']):.6f} m (n={len(data['T'])})" if data['T'] else "  T (ATE):       N/A")
    print(f"  R:             {np.mean(data['R']):.6f} (n={len(data['R'])})" if data['R'] else "  R:             N/A")
    print(f"  PSNR:          {np.mean(data['PSNR']):.2f} dB (n={len(data['PSNR'])})" if data['PSNR'] else "  PSNR:          N/A")
    print(f"  SSIM:          {np.mean(data['SSIM']):.4f} (n={len(data['SSIM'])})" if data['SSIM'] else "  SSIM:          N/A")
    print(f"  LPIPS:         {np.mean(data['LPIPS']):.4f} (n={len(data['LPIPS'])})" if data['LPIPS'] else "  LPIPS:         N/A")
    print(f"  Tracking FPS:  {np.mean(data['Tracking_FPS']):.2f} (n={len(data['Tracking_FPS'])})" if data['Tracking_FPS'] else "  Tracking FPS:  N/A")
    print(f"  Rendering FPS: {np.mean(data['Rendering_FPS']):.2f} (n={len(data['Rendering_FPS'])})" if data['Rendering_FPS'] else "  Rendering FPS: N/A")
    print(f"  T_std:         {np.mean(data['T_std']):.6f} m (n={len(data['T_std'])})" if data['T_std'] else "  T_std:         N/A")

print("\n" + "="*80)
