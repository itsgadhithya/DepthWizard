"""Metadata extractor for EXIF, GPS, and GeoTIFF tags with explicit provenance tracking."""

import io
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
from PIL import Image, ExifTags
import numpy as np

from backend.models.metadata import (
    ImageMetadata,
    ExifMetadata,
    GPSMetadata,
    GeoTIFFMetadata,
    MetadataFieldStatus,
)


class MetadataExtractor:
    """Extracts all available metadata from image without fabricating missing tags."""

    @classmethod
    def extract_metadata(
        cls,
        image_input: Union[str, Path, bytes, io.BytesIO],
        filename: str = "image.jpg",
        format_hint: Optional[str] = None,
    ) -> ImageMetadata:
        """Extract all available EXIF, GPS, and GeoTIFF metadata from the input."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            filename = path.name
            pil_img = Image.open(path)
            file_bytes = None
            file_path_str = str(path)
        else:
            if isinstance(image_input, bytes):
                file_bytes = image_input
                pil_img = Image.open(io.BytesIO(file_bytes))
            else:
                image_input.seek(0)
                file_bytes = image_input.read()
                pil_img = Image.open(io.BytesIO(file_bytes))
            file_path_str = None

        width, height = pil_img.size
        img_format = (format_hint or pil_img.format or "JPEG").upper()

        metadata = ImageMetadata(
            filename=filename,
            format=img_format,
            width=width,
            height=height,
            channels=len(pil_img.getbands()),
            bit_depth=8,  # Default, adjusted below
        )

        metadata.record_field("dimensions", MetadataFieldStatus.PRESENT, source="image_header", confidence=1.0)
        metadata.record_field("format", MetadataFieldStatus.PRESENT, source="image_header", confidence=1.0)

        # 1. Extract EXIF
        exif_obj, gps_obj = cls._parse_exif(pil_img, metadata)
        metadata.exif = exif_obj
        metadata.gps = gps_obj
        metadata.has_exif = exif_obj is not None
        metadata.has_gps = gps_obj is not None and (gps_obj.latitude is not None and gps_obj.longitude is not None)

        # 2. Extract GeoTIFF if applicable
        geotiff_obj = cls._parse_geotiff(file_path_str, file_bytes, metadata)
        metadata.geotiff = geotiff_obj
        metadata.has_geotiff = geotiff_obj is not None

        if metadata.has_geotiff:
            metadata.format = "GeoTIFF"

        return metadata

    @classmethod
    def _parse_exif(
        cls,
        pil_img: Image.Image,
        meta: ImageMetadata,
    ) -> Tuple[Optional[ExifMetadata], Optional[GPSMetadata]]:
        """Parse standard EXIF and GPS tags."""
        raw_exif = None
        try:
            raw_exif = pil_img._getexif()
        except Exception:
            pass

        if not raw_exif:
            meta.record_field("exif", MetadataFieldStatus.ABSENT, source="none", confidence=1.0, notes="No EXIF header found")
            meta.record_field("gps", MetadataFieldStatus.ABSENT, source="none", confidence=1.0, notes="No GPS data found")
            return None, None

        # Map tag IDs to tag names
        tag_dict: Dict[str, Any] = {}
        for tag_id, value in raw_exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            tag_dict[tag_name] = value

        exif = ExifMetadata()
        gps = GPSMetadata()

        # Camera make / model
        if "Make" in tag_dict:
            exif.make = str(tag_dict["Make"]).strip()
            meta.record_field("camera_make", MetadataFieldStatus.PRESENT, source="exif", confidence=1.0)
        else:
            meta.record_field("camera_make", MetadataFieldStatus.ABSENT, source="none", confidence=1.0)

        if "Model" in tag_dict:
            exif.model = str(tag_dict["Model"]).strip()
            meta.record_field("camera_model", MetadataFieldStatus.PRESENT, source="exif", confidence=1.0)
        else:
            meta.record_field("camera_model", MetadataFieldStatus.ABSENT, source="none", confidence=1.0)

        # Focal length
        if "FocalLength" in tag_dict:
            try:
                fl = tag_dict["FocalLength"]
                exif.focal_length_mm = float(fl)
                meta.record_field("focal_length_mm", MetadataFieldStatus.PRESENT, source="exif", confidence=1.0)
            except Exception:
                meta.record_field("focal_length_mm", MetadataFieldStatus.ABSENT, source="none", confidence=1.0)
        else:
            meta.record_field("focal_length_mm", MetadataFieldStatus.ABSENT, source="none", confidence=1.0)

        if "FocalLengthIn35mmFilm" in tag_dict:
            try:
                exif.focal_length_35mm_equiv = float(tag_dict["FocalLengthIn35mmFilm"])
                meta.record_field("focal_length_35mm_equiv", MetadataFieldStatus.PRESENT, source="exif", confidence=1.0)
            except Exception:
                meta.record_field("focal_length_35mm_equiv", MetadataFieldStatus.ABSENT, source="none", confidence=1.0)

        # Exposure, ISO, F-number
        if "ISOSpeedRatings" in tag_dict:
            try:
                exif.iso = int(tag_dict["ISOSpeedRatings"])
            except Exception:
                pass
        if "FNumber" in tag_dict:
            try:
                exif.f_number = float(tag_dict["FNumber"])
            except Exception:
                pass
        if "ExposureTime" in tag_dict:
            try:
                exif.exposure_time = float(tag_dict["ExposureTime"])
            except Exception:
                pass
        if "Orientation" in tag_dict:
            try:
                exif.orientation = int(tag_dict["Orientation"])
            except Exception:
                pass
        if "DateTime" in tag_dict or "DateTimeOriginal" in tag_dict:
            exif.timestamp = str(tag_dict.get("DateTimeOriginal") or tag_dict.get("DateTime"))

        # GPS Extraction
        gps_info = tag_dict.get("GPSInfo")
        if gps_info:
            gps_tags = {}
            for k, v in gps_info.items():
                name = ExifTags.GPSTAGS.get(k, str(k))
                gps_tags[name] = v

            lat = cls._convert_gps_coords(gps_tags.get("GPSLatitude"), gps_tags.get("GPSLatitudeRef"))
            lon = cls._convert_gps_coords(gps_tags.get("GPSLongitude"), gps_tags.get("GPSLongitudeRef"))
            alt = cls._convert_gps_altitude(gps_tags.get("GPSAltitude"), gps_tags.get("GPSAltitudeRef"))

            if lat is not None and lon is not None:
                gps.latitude = lat
                gps.longitude = lon
                meta.record_field("gps_lat_lon", MetadataFieldStatus.PRESENT, source="exif_gps", confidence=1.0)
            else:
                meta.record_field("gps_lat_lon", MetadataFieldStatus.ABSENT, source="none", confidence=1.0)

            if alt is not None:
                gps.altitude = alt
                alt_ref_raw = gps_tags.get("GPSAltitudeRef", 0)
                if isinstance(alt_ref_raw, bytes):
                    gps.altitude_ref = int(alt_ref_raw[0]) if len(alt_ref_raw) > 0 else 0
                else:
                    try:
                        gps.altitude_ref = int(alt_ref_raw)
                    except Exception:
                        gps.altitude_ref = 0
                meta.record_field("gps_altitude", MetadataFieldStatus.PRESENT, source="exif_gps", confidence=1.0)
            else:
                meta.record_field("gps_altitude", MetadataFieldStatus.ABSENT, source="none", confidence=1.0)

            if "GPSDOP" in gps_tags:
                try:
                    gps.dop = float(gps_tags["GPSDOP"])
                except Exception:
                    pass
        else:
            meta.record_field("gps", MetadataFieldStatus.ABSENT, source="none", confidence=1.0, notes="No GPSInfo in EXIF")

        return exif, gps

    @classmethod
    def _convert_gps_coords(cls, coords: Any, ref: Optional[str]) -> Optional[float]:
        """Convert degrees, minutes, seconds rational tuple to decimal degrees."""
        if not coords or len(coords) < 3:
            return None
        try:
            d = float(coords[0])
            m = float(coords[1])
            s = float(coords[2])
            dec = d + (m / 60.0) + (s / 3600.0)
            ref_str = ref.decode("ascii", errors="ignore") if isinstance(ref, bytes) else str(ref or "")
            if ref_str.upper() in ("S", "W"):
                dec = -dec
            return dec
        except Exception:
            return None

    @classmethod
    def _convert_gps_altitude(cls, altitude: Any, ref: Any) -> Optional[float]:
        """Convert GPS altitude rational to float meters."""
        if altitude is None:
            return None
        try:
            alt = float(altitude)
            # Ref 1 means below sea level
            is_below = False
            if isinstance(ref, bytes):
                is_below = (len(ref) > 0 and ref[0] == 1) or ref == b"1"
            else:
                try:
                    is_below = int(ref) == 1
                except Exception:
                    is_below = False
            if is_below:
                alt = -alt
            return alt
        except Exception:
            return None

    @classmethod
    def _parse_geotiff(
        cls,
        file_path: Optional[str],
        file_bytes: Optional[bytes],
        meta: ImageMetadata,
    ) -> Optional[GeoTIFFMetadata]:
        """Extract geospatial tags from GeoTIFF using rasterio."""
        try:
            import rasterio
            from rasterio.io import MemoryFile

            if file_path:
                with rasterio.open(file_path) as src:
                    return cls._extract_rasterio_meta(src, meta)
            elif file_bytes:
                with MemoryFile(file_bytes) as memfile:
                    with memfile.open() as src:
                        return cls._extract_rasterio_meta(src, meta)
        except Exception:
            pass

        meta.record_field("geotiff", MetadataFieldStatus.ABSENT, source="none", confidence=1.0)
        return None

    @classmethod
    def _extract_rasterio_meta(cls, src: Any, meta: ImageMetadata) -> Optional[GeoTIFFMetadata]:
        """Extract rasterio dataset metadata and determine if raster represents a DEM."""
        if not src.crs:
            meta.record_field("geotiff", MetadataFieldStatus.ABSENT, source="none", confidence=1.0, notes="No CRS found in TIFF")
            return None

        crs_str = src.crs.to_string()
        transform_list = list(src.transform)[:6]
        bounds_list = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
        res_list = [abs(src.res[0]), abs(src.res[1])]
        nodata = float(src.nodata) if src.nodata is not None else None
        band_count = src.count
        dtype_str = src.dtypes[0] if len(src.dtypes) > 0 else "uint8"

        is_dem = False
        min_elev: Optional[float] = None
        max_elev: Optional[float] = None
        mean_elev: Optional[float] = None
        std_elev: Optional[float] = None

        # Inspect if single-band raster is an elevation model
        if band_count == 1:
            try:
                band_data = src.read(1).astype(np.float32)
                valid_mask = np.isfinite(band_data)
                if nodata is not None:
                    valid_mask = valid_mask & (band_data != nodata)

                if np.any(valid_mask):
                    valid_vals = band_data[valid_mask]
                    min_elev = round(float(np.min(valid_vals)), 3)
                    max_elev = round(float(np.max(valid_vals)), 3)
                    mean_elev = round(float(np.mean(valid_vals)), 3)
                    std_elev = round(float(np.std(valid_vals)), 3)

                    # Plausible elevation bounds check (e.g. -500m to 9000m on Earth)
                    if (
                        dtype_str.startswith("float")
                        or dtype_str.startswith("int")
                        or dtype_str.startswith("uint16")
                    ) and (-500.0 <= min_elev <= 9000.0) and (-500.0 <= max_elev <= 9000.0):
                        is_dem = True
            except Exception:
                pass

        meta.is_dem = is_dem

        geo = GeoTIFFMetadata(
            crs=crs_str,
            transform=transform_list,
            bounds=bounds_list,
            resolution=res_list,
            nodata=nodata,
            driver=src.driver,
            band_count=band_count,
            dtype=dtype_str,
            is_dem=is_dem,
            min_elevation_m=min_elev,
            max_elevation_m=max_elev,
            mean_elevation_m=mean_elev,
            std_elevation_m=std_elev,
        )

        meta.record_field("geotiff_crs", MetadataFieldStatus.PRESENT, source="geotiff", confidence=1.0, notes=crs_str)
        meta.record_field("geotiff_transform", MetadataFieldStatus.PRESENT, source="geotiff", confidence=1.0)
        meta.record_field("geotiff_bounds", MetadataFieldStatus.PRESENT, source="geotiff", confidence=1.0)
        meta.record_field("is_dem", MetadataFieldStatus.INFERRED, source="geotiff_analysis", confidence=0.95 if is_dem else 0.85, notes=f"is_dem={is_dem}, bands={band_count}, dtype={dtype_str}")

        return geo
