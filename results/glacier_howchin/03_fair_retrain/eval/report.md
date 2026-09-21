# Fair re-evaluation (same 73 held-out views, same reference photos)

Views: 73 (with sky removed by the masks: 34). Keep region eroded 4px.

| Variant | naive PSNR/SSIM/LPIPS (own GT, gsplat_eval) | Full frame PSNR / SSIM / LPIPS | Keep region PSNR / SSIM / LPIPS | Sky region PSNR |
|---|---|---|---|---|
| No mask (raw imagery) | 22.08 / 0.766 / 0.315 | 22.08 / 0.766 / 0.315 | 22.02 / 0.749 / 0.260 | 22.89 |
| SAM3, no agent | 23.41 / 0.773 / 0.263 | 11.33 / 0.657 / 0.350 | 23.06 / 0.754 / 0.256 | 2.59 |
| Agent + validation | 23.28 / 0.771 / 0.260 | 11.38 / 0.659 / 0.350 | 23.02 / 0.753 / 0.256 | 2.64 |

PSNR = pooled over all pixels of the region; SSIM/LPIPS = mean over views.

## Paired differences (mean over views, 95% bootstrap CI)

| Comparison | Region | Views | dPSNR (dB) | dSSIM | dLPIPS |
|---|---|---|---|---|---|
| Agent + validation - No mask (raw imagery) | keep | all | +0.300 [-0.018, +0.652] | +0.004 [-0.001, +0.011] | -0.004 [-0.011, -0.000] |
| Agent + validation - No mask (raw imagery) | keep | sky_views | +0.523 [-0.146, +1.252] | +0.008 [-0.002, +0.023] | -0.009 [-0.024, -0.000] |
| Agent + validation - No mask (raw imagery) | keep | no_sky_views | +0.105 [-0.011, +0.234] | -0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.001] |
| Agent + validation - No mask (raw imagery) | full | all | -6.966 [-8.880, -5.116] | -0.107 [-0.138, -0.077] | +0.035 [+0.022, +0.047] |
| Agent + validation - No mask (raw imagery) | full | sky_views | -15.077 [-16.698, -13.446] | -0.230 [-0.262, -0.196] | +0.075 [+0.054, +0.091] |
| Agent + validation - No mask (raw imagery) | full | no_sky_views | +0.105 [-0.013, +0.233] | -0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.001] |
| Agent + validation - SAM3, no agent | keep | all | -0.019 [-0.128, +0.089] | -0.001 [-0.002, +0.000] | +0.000 [-0.000, +0.001] |
| Agent + validation - SAM3, no agent | keep | sky_views | -0.067 [-0.227, +0.085] | -0.001 [-0.003, -0.000] | +0.001 [-0.000, +0.001] |
| Agent + validation - SAM3, no agent | keep | no_sky_views | +0.022 [-0.125, +0.179] | +0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.001] |
| Agent + validation - SAM3, no agent | full | all | +0.039 [-0.041, +0.122] | +0.002 [+0.000, +0.003] | -0.001 [-0.001, +0.000] |
| Agent + validation - SAM3, no agent | full | sky_views | +0.058 [+0.033, +0.084] | +0.003 [+0.001, +0.005] | -0.001 [-0.002, -0.000] |
| Agent + validation - SAM3, no agent | full | no_sky_views | +0.022 [-0.130, +0.178] | +0.000 [-0.001, +0.001] | -0.000 [-0.001, +0.001] |
| SAM3, no agent - No mask (raw imagery) | keep | all | +0.319 [-0.009, +0.697] | +0.005 [-0.000, +0.012] | -0.005 [-0.011, -0.000] |
| SAM3, no agent - No mask (raw imagery) | keep | sky_views | +0.590 [-0.080, +1.350] | +0.010 [-0.001, +0.025] | -0.010 [-0.024, -0.001] |
| SAM3, no agent - No mask (raw imagery) | keep | no_sky_views | +0.083 [-0.086, +0.257] | -0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.001] |
| SAM3, no agent - No mask (raw imagery) | full | all | -7.004 [-8.890, -5.135] | -0.109 [-0.140, -0.079] | +0.035 [+0.023, +0.047] |
| SAM3, no agent - No mask (raw imagery) | full | sky_views | -15.134 [-16.733, -13.461] | -0.233 [-0.265, -0.200] | +0.076 [+0.055, +0.092] |
| SAM3, no agent - No mask (raw imagery) | full | no_sky_views | +0.083 [-0.091, +0.252] | -0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.001] |
