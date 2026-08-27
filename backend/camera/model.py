"""Camera model builder computing intrinsic and extrinsic parameters with provenance."""

import math
from typing import Optional, Dict, Any
from backend.models.metadata import ImageMetadata, MetadataFieldStatus, FieldProvenance
from backend.models.camera import CameraIntrinsics, CameraExtrinsics, CameraModel
from backend.camera.sensor_db import lookup_sensor_dimensions


class CameraModelBuilder:
    """Constructs explicit CameraModel with provenance tracking."""

    @classmethod
    def build_camera_model(
        cls,
        metadata: ImageMetadata,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> CameraModel:
        """Construct CameraModel from metadata and optional user overrides.

        Args:
            metadata: Extracted ImageMetadata.
            overrides: Optional user dictionary (e.g. {'fx': 1200, 'fy': 1200, 'sensor_width_mm': 13.2}).

        Returns:
            CameraModel with intrinsics, extrinsics, and provenance.
        """
        overrides = overrides or {}
        w = metadata.width
        h = metadata.height
        provenance: Dict[str, FieldProvenance] = {}

        # 1. Intrinsics computation
        fx: Optional[float] = overrides.get("fx")
        fy: Optional[float] = overrides.get("fy")
        cx: float = float(overrides.get("cx", w / 2.0))
        cy: float = float(overrides.get("cy", h / 2.0))
        k1: float = float(overrides.get("k1", 0.0))
        k2: float = float(overrides.get("k2", 0.0))
        p1: float = float(overrides.get("p1", 0.0))
        p2: float = float(overrides.get("p2", 0.0))
        k3: float = float(overrides.get("k3", 0.0))

        is_estimated = True
        estimation_method = "heuristic_fov"
        confidence = 0.35

        # Check explicit focal overrides
        if fx is not None and fy is not None:
            is_estimated = False
            estimation_method = "calibrated_user_input"
            confidence = 1.0
            provenance["focal_length"] = FieldProvenance(
                field_name="focal_length",
                status=MetadataFieldStatus.PRESENT,
                source="user_override",
                confidence=1.0,
                notes=f"User calibrated fx={fx}, fy={fy}",
            )
        else:
            # Check EXIF focal length + sensor database
            focal_mm = overrides.get("focal_length_mm")
            sensor_w_mm = overrides.get("sensor_width_mm")

            if focal_mm is None and metadata.exif and metadata.exif.focal_length_mm:
                focal_mm = metadata.exif.focal_length_mm

            if sensor_w_mm is None and metadata.exif:
                sensor_dims = lookup_sensor_dimensions(metadata.exif.make, metadata.exif.model)
                if sensor_dims:
                    sensor_w_mm = sensor_dims[0]

            if focal_mm and sensor_w_mm and sensor_w_mm > 0:
                # Compute pixel focal length
                fx = float(focal_mm * w / sensor_w_mm)
                fy = fx
                is_estimated = False
                estimation_method = "exif_sensor_lookup"
                confidence = 0.90
                provenance["focal_length"] = FieldProvenance(
                    field_name="focal_length",
                    status=MetadataFieldStatus.INFERRED,
                    source="exif_focal_and_sensor_db",
                    confidence=0.90,
                    notes=f"f={focal_mm}mm, sensor_w={sensor_w_mm}mm -> fx={fx:.1f}px",
                )
            elif metadata.exif and metadata.exif.focal_length_35mm_equiv:
                # Compute from 35mm equivalent (36mm standard width)
                f_35 = metadata.exif.focal_length_35mm_equiv
                diag_px = math.sqrt(w**2 + h**2)
                diag_35mm = math.sqrt(36.0**2 + 24.0**2)  # 43.27mm
                fx = float(f_35 * diag_px / diag_35mm)
                fy = fx
                is_estimated = True
                estimation_method = "exif_35mm_equiv"
                confidence = 0.75
                provenance["focal_length"] = FieldProvenance(
                    field_name="focal_length",
                    status=MetadataFieldStatus.INFERRED,
                    source="exif_35mm_equivalent",
                    confidence=0.75,
                    notes=f"f_35mm={f_35}mm -> fx={fx:.1f}px",
                )
            else:
                # Heuristic estimation: Standard 65 degree horizontal field of view
                hfov_rad = math.radians(65.0)
                fx = float(w / (2.0 * math.tan(hfov_rad / 2.0)))
                fy = fx
                is_estimated = True
                estimation_method = "heuristic_fov"
                confidence = 0.35
                provenance["focal_length"] = FieldProvenance(
                    field_name="focal_length",
                    status=MetadataFieldStatus.ESTIMATED,
                    source="heuristic_fov_65deg",
                    confidence=0.35,
                    notes="No focal length found in metadata. Estimated assuming standard 65 deg HFOV.",
                )

        intrinsics = CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            k1=k1,
            k2=k2,
            p1=p1,
            p2=p2,
            k3=k3,
            width=w,
            height=h,
            is_estimated=is_estimated,
            estimation_method=estimation_method,
            confidence=confidence,
        )

        # 2. Extrinsics computation
        extrinsics: Optional[CameraExtrinsics] = None
        has_pos = False
        has_ori = False

        # Check Position sources: EXIF GPS or user overrides
        lat = overrides.get("latitude")
        lon = overrides.get("longitude")
        alt = overrides.get("altitude_m")

        if lat is None and metadata.has_gps and metadata.gps:
            lat = metadata.gps.latitude
        if lon is None and metadata.has_gps and metadata.gps:
            lon = metadata.gps.longitude
        if alt is None and metadata.has_gps and metadata.gps:
            alt = metadata.gps.altitude

        pos_x = overrides.get("position_x")
        pos_y = overrides.get("position_y")
        pos_z = overrides.get("position_z", alt)
        proj_crs = overrides.get("projected_crs")

        if lat is not None and lon is not None:
            has_pos = True
            # Convert geographic coordinates to projected UTM CRS if pos_x/pos_y not directly provided
            if pos_x is None or pos_y is None:
                try:
                    from backend.geospatial.crs import CRSHelper
                    from pyproj import Transformer
                    proj_crs = proj_crs or CRSHelper.get_utm_crs_for_latlon(lon, lat)
                    transformer = Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
                    pos_x, pos_y = transformer.transform(lon, lat)
                except Exception:
                    pos_x, pos_y = None, None

            pos_source = "user_override" if ("latitude" in overrides or "position_x" in overrides) else "exif_gps"
            provenance["camera_position"] = FieldProvenance(
                field_name="camera_position",
                status=MetadataFieldStatus.PRESENT,
                source=pos_source,
                confidence=1.0 if pos_source == "user_override" else 0.95,
                notes=f"Lat={lat:.6f}, Lon={lon:.6f}, Alt={alt}m (Projected: {proj_crs} Easting={pos_x:.2f}m, Northing={pos_y:.2f}m)" if pos_x else f"Lat={lat:.6f}, Lon={lon:.6f}, Alt={alt}m",
            )
        elif pos_x is not None and pos_y is not None:
            has_pos = True
            provenance["camera_position"] = FieldProvenance(
                field_name="camera_position",
                status=MetadataFieldStatus.PRESENT,
                source="user_override",
                confidence=1.0,
                notes=f"Projected X={pos_x}, Y={pos_y}, Z={pos_z}",
            )
        else:
            provenance["camera_position"] = FieldProvenance(
                field_name="camera_position",
                status=MetadataFieldStatus.ABSENT,
                source="none",
                confidence=1.0,
                notes="No GPS or spatial position coordinates found in metadata.",
            )

        # Check Orientation sources (explicit overrides or genuine orientation metadata)
        # CRITICAL RULE: Never fabricate yaw/pitch/roll from GPS coordinates alone.
        yaw = overrides.get("yaw_deg")
        pitch = overrides.get("pitch_deg")
        roll = overrides.get("roll_deg")

        if yaw is not None and pitch is not None and roll is not None:
            has_ori = True
            provenance["camera_orientation"] = FieldProvenance(
                field_name="camera_orientation",
                status=MetadataFieldStatus.PRESENT,
                source="user_override",
                confidence=1.0,
                notes=f"User calibrated orientation: Yaw={yaw:.1f}deg, Pitch={pitch:.1f}deg, Roll={roll:.1f}deg",
            )
        else:
            has_ori = False
            provenance["camera_orientation"] = FieldProvenance(
                field_name="camera_orientation",
                status=MetadataFieldStatus.ABSENT,
                source="none",
                confidence=1.0,
                notes="No camera orientation (yaw/pitch/roll) found in metadata or overrides. Orientation remains unknown.",
            )

        has_complete_pose = has_pos and has_ori

        if has_pos or has_ori:
            extrinsics = CameraExtrinsics(
                position_x=pos_x,
                position_y=pos_y,
                position_z=pos_z,
                latitude=lat,
                longitude=lon,
                altitude_m=alt,
                projected_crs=proj_crs,
                yaw_deg=yaw,
                pitch_deg=pitch,
                roll_deg=roll,
                is_position_available=has_pos,
                is_orientation_available=has_ori,
                is_complete_pose_available=has_complete_pose,
                is_estimated=not has_complete_pose,
                coordinate_frame="WGS84_ENU" if proj_crs is None else f"{proj_crs}_ENU",
            )

        return CameraModel(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            has_position=has_pos,
            has_orientation=has_ori,
            has_complete_pose=has_complete_pose,
            provenance=provenance,
        )
