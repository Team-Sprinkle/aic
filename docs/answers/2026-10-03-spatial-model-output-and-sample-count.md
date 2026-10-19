# What the spatial SC model predicts and how many samples it used

**It is a separate identity/in-frame head, not the port-pose predictor.** Given **one camera image**, the requested SC port code, camera identity and measured TCP pose, it outputs **one number**: the probability that the requested port's opening projects into that camera image. It does not output the opening's pixel coordinates, 3D position or orientation. Its label is geometric *in-frame projection*, not verified visual identifiability; an occluded port can still project into the image.

The existing pose pipeline has other components: a frozen coarse locator proposes the selected opening's pixel location, an original-resolution crop refiner improves the pixel point, and the camera views are combined to estimate a 3D point. A separate orientation model is used for orientation. The new head uses the **same frozen RGB backbone features** as the coarse locator but has its **own trained layers and checkpoint**. It was tested alongside the pose components, but it has **not yet been integrated into the live transport-to-insertion switch**.

```text
Each wrist-camera RGB + requested port code + measured TCP
                        |
             frozen image backbone
                        |
              +---------+----------+
              |                    |
     new spatial identity head   existing pixel locator
     "is this requested port    + native crop refiner
      in this camera image?"     "where is the opening?"
              |                    |
              +-- three-camera ----+--> proposed selected-port estimate
                   checks
```

**Training count:** 2,358 **individual camera images** from 95 complete reset groups:

| Source | Reset groups | Single-camera training images |
| --- | ---: | ---: |
| One-port SC | 12 (8 new, 4 older) | 864 |
| Two-port SC | 83 | 1,494 |
| **Total** | **95** | **2,358** |

Each image was queried twice in training: once for “port 0?” and once for “port 1?”. Thus there were **4,716 distinct image–requested-port labeled pairs** before minibatch resampling. The fit ran **800 optimizer updates**; each update sampled 32 one-port and 32 two-port camera images, then asked both port questions for each image. These repeats are not additional unique data.

**No, one training sample was not a three-image bundle.** The model trained on one camera image at a time, with a camera-ID input. At evaluation, we ran it separately on the **three camera images** at one decision time and accepted the requested port only if **all three** outputs were positive. The 109 decisions in the previous table are three-camera decision times from **four held-out reset groups**, not 109 independent episodes. The held-out one-port development set had 288 sampled individual camera images; the original two-port validation set had 378. Both were excluded from fitting.

The [training summary](/home/nas/chmin/aic_archives/2026-10-02_selected_port_handoff/sc_spatial_identity_probe_v2/summary.json), [model code](../../scripts/train_sc_spatial_identity_probe.py), and [experiment record](../experiments/2026-10-02-sc-absent-port-identity.md#spatial-selected-port-continuation-on-new-reset-groups) retain the exact counts and split.
