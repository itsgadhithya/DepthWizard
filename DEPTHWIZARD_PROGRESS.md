# DepthWizard — Implementation Progress

## Completed

- [x] Backend foundation
- [x] Image ingestion
- [x] Metadata extraction
- [x] DepthAnything V2 integration

## Currently Working On

- [ ] Metric calibration

## Next

- [ ] Georeferencing
- [ ] DSM generation
- [ ] Validation
- [ ] 3D visualization

## Important Decisions

- Current product accepts one image.
- DepthAnything V2 is the primary monocular depth estimator.
- Relative depth must never be represented as metric depth without calibration.
- GPS coordinates alone are insufficient for metric depth.
- Multi-image SfM/MVS is reserved for a future version.