"""
Benchmark: GaussianRasterizer time vs Gaussian count.
Uses one existing frame PLY from the HUGS output.
Run with: conda activate hugs && python benchmark_gaussians.py
"""

import math, time
import numpy as np
import torch
import torch.nn.functional as F

# ── parse PLY without plyfile ──────────────────────────────────────────────
def load_ply(path):
    with open(path, 'rb') as f:
        # read header
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
        n_verts = int([l for l in header if l.startswith('element vertex')][0].split()[-1])
        props = [l.split()[-1] for l in header if l.startswith('property float')]
        n_props = len(props)
        data = np.frombuffer(f.read(n_verts * n_props * 4),
                             dtype=np.float32).reshape(n_verts, n_props)
    prop_idx = {name: i for i, name in enumerate(props)}
    return data, prop_idx, n_verts

# ── dummy camera ──────────────────────────────────────────────────────────
def make_camera(H=540, W=960):
    fovx = fovy = math.radians(60)

    # View matrix: camera sitting at (0,0,5) looking toward -Z
    view = torch.eye(4, device='cuda')
    view[2, 3] = -5.0          # translate world so scene is in front of camera

    # Projection matrix built from FOV (OpenGL-style perspective)
    near, far = 0.01, 100.0
    f = 1.0 / math.tan(fovx * 0.5)
    proj = torch.zeros(4, 4, device='cuda')
    proj[0, 0] = f
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = 1.0
    full_proj = proj @ view

    return {
        'image_height': H, 'image_width': W,
        'fovx': fovx, 'fovy': fovy,
        'world_view_transform': view,
        'full_proj_transform':  full_proj,
        'camera_center':        torch.tensor([0., 0., 5.], device='cuda'),
    }

# ── time one rasterizer call ───────────────────────────────────────────────
def time_render(means3D, colors, opacity, scales, rotations, data, n_warmup=5, n_runs=20):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    bg = torch.zeros(3, device='cuda')
    tanfovx = math.tan(data['fovx'] * 0.5)
    tanfovy = math.tan(data['fovy'] * 0.5)
    settings = GaussianRasterizationSettings(
        image_height=int(data['image_height']),
        image_width=int(data['image_width']),
        tanfovx=tanfovx, tanfovy=tanfovy, bg=bg,
        scale_modifier=1.0,
        viewmatrix=data['world_view_transform'],
        projmatrix=data['full_proj_transform'],
        sh_degree=0, campos=data['camera_center'],
        prefiltered=False, debug=False,
    )
    rasterizer = GaussianRasterizer(settings)

    def one_pass():
        means2D = torch.zeros_like(means3D, requires_grad=True)
        rasterizer(means3D=means3D, means2D=means2D,
                   colors_precomp=colors,   # skip SH evaluation entirely
                   opacities=opacity, scales=scales, rotations=rotations)

    # warmup
    for _ in range(n_warmup):
        one_pass()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        one_pass()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000
    return elapsed_ms

# ── main ───────────────────────────────────────────────────────────────────
PLY = ('output/human_scene/neuman/bike/hugs_trimlp/'
       'demo-dataset.seq=bike/2026-06-05_10-17-00/anim_ply/00091_splat.ply')

data_np, pidx, total = load_ply(PLY)
print(f"Loaded {total:,} Gaussians from PLY")

# Print value ranges so we can verify activations are applied correctly
for col in ['scale_0', 'opacity', 'x', 'z']:
    v = data_np[:, pidx[col]]
    print(f"  {col:10s}: min={v.min():.3f}  max={v.max():.3f}  mean={v.mean():.3f}")

# PLY stores log(scale) and inverse_sigmoid(opacity) — apply activations
# (see hugs/utils/vis.py save_posed_ply lines 72-75)
raw_opac  = torch.tensor(data_np[:, pidx['opacity']], dtype=torch.float32)
raw_scale = torch.tensor(data_np[:, [pidx['scale_0'], pidx['scale_1'], pidx['scale_2']]],
                         dtype=torch.float32)
opac_act  = torch.sigmoid(raw_opac).unsqueeze(1).cuda()      # (N,1)
scale_act = torch.exp(raw_scale).cuda()                       # (N,3)

print(f"\nAfter activations:")
print(f"  opacity : min={opac_act.min():.3f}  max={opac_act.max():.3f}")
print(f"  scale_0 : min={scale_act[:,0].min():.5f}  max={scale_act[:,0].max():.5f}")

xyz_all = torch.tensor(data_np[:, [pidx['x'], pidx['y'], pidx['z']]],
                       dtype=torch.float32).cuda()
rot_all = F.normalize(
    torch.tensor(data_np[:, [pidx['rot_0'], pidx['rot_1'],
                              pidx['rot_2'], pidx['rot_3']]],
                 dtype=torch.float32), dim=-1).cuda()

# Sort by opacity descending so pruned subsets keep the most visible Gaussians
order = torch.argsort(opac_act.squeeze(), descending=True)
xyz_all   = xyz_all[order]
opac_act  = opac_act[order]
scale_act = scale_act[order]
rot_all   = rot_all[order]

# Centre the scene: put camera above the centroid looking toward -Z
centroid = xyz_all.mean(dim=0)
cam = make_camera()
offset = centroid.clone()
offset[2] += 3.0   # camera 3 units behind the scene centroid
cam['camera_center'] = offset
cam['world_view_transform'][0, 3] = -centroid[0]
cam['world_view_transform'][1, 3] = -centroid[1]
cam['world_view_transform'][2, 3] = -(centroid[2] + 3.0)

# target counts — adjust to what the scene actually has
targets = [50_000, 100_000, 200_000, total]
targets = sorted(set(t for t in targets if t <= total))

print(f"\n{'Gaussians':>12}  {'ms/frame':>10}  {'FPS':>8}")
print('-' * 34)
results = []
for n in targets:
    xyz   = xyz_all[:n].contiguous()
    opac  = opac_act[:n].contiguous()
    sc    = scale_act[:n].contiguous()
    rot   = rot_all[:n].contiguous()
    colors = torch.full((n, 3), 0.5, device='cuda')   # flat grey — no SH needed

    ms = time_render(xyz, colors, opac, sc, rot, cam)
    fps = 1000 / ms
    results.append((n, ms, fps))
    print(f"{n:>12,}  {ms:>10.2f}  {fps:>8.1f}")

# ── plot ───────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

counts = [r[0] for r in results]
ms_vals = [r[1] for r in results]
fps_vals = [r[2] for r in results]

fig, ax1 = plt.subplots(figsize=(8, 5))

color_ms  = '#2196F3'
color_fps = '#F44336'

ax1.plot(counts, ms_vals, 'o-', color=color_ms, linewidth=2, markersize=6, label='ms/frame')
ax1.set_xlabel('Number of Gaussians')
ax1.set_ylabel('Render time (ms/frame)', color=color_ms)
ax1.tick_params(axis='y', labelcolor=color_ms)
ax1.set_xticks(counts)
ax1.set_xticklabels([f'{c:,}' for c in counts], rotation=15)

ax2 = ax1.twinx()
ax2.plot(counts, fps_vals, 's--', color=color_fps, linewidth=2, markersize=6, label='FPS')
ax2.set_ylabel('FPS', color=color_fps)
ax2.tick_params(axis='y', labelcolor=color_fps)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('GaussianRasterizer: render time & FPS vs Gaussian count')
plt.tight_layout()
out = 'benchmark_gaussians.png'
plt.savefig(out, dpi=150)
print(f"\nPlot saved to {out}")
