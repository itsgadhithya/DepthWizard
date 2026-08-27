"""Geospatial CRS management and UTM zone auto-detection."""

import math
from typing import Optional, Tuple
import pyproj
from pyproj import CRS

from backend.models.geospatial import CRSInfo


class CRSHelper:
    """Utilities for CRS validation, conversion, and UTM projection resolution."""

    @classmethod
    def get_utm_crs_for_latlon(cls, lon: float, lat: float) -> str:
        """Calculate the appropriate EPSG code for UTM projected coordinate system.

        Args:
            lon: Longitude in decimal degrees [-180, 180].
            lat: Latitude in decimal degrees [-90, 90].

        Returns:
            EPSG string, e.g. 'EPSG:32643' (North) or 'EPSG:32743' (South).
        """
        # Constrain lon to [-180, 180)
        norm_lon = (lon + 180.0) % 360.0 - 180.0
        zone = int(math.floor((norm_lon + 180.0) / 6.0)) + 1
        zone = max(1, min(60, zone))

        if lat >= 0:
            epsg_code = 32600 + zone  # WGS 84 / UTM Northern Hemisphere
        else:
            epsg_code = 32700 + zone  # WGS 84 / UTM Southern Hemisphere

        return f"EPSG:{epsg_code}"

    @classmethod
    def parse_crs_info(cls, crs_str: str) -> CRSInfo:
        """Inspect and parse a CRS string into typed CRSInfo."""
        try:
            crs_obj = CRS.from_user_input(crs_str)
            is_proj = crs_obj.is_projected
            is_geo = crs_obj.is_geographic
            datum = crs_obj.datum.name if crs_obj.datum else "Unknown"

            # Check if UTM
            utm_zone = None
            utm_hemi = None
            if is_proj and "utm" in crs_obj.name.lower():
                try:
                    utm_zone = crs_obj.utm_zone
                except Exception:
                    pass

            return CRSInfo(
                crs_string=crs_obj.to_string(),
                is_projected=is_proj,
                is_geographic=is_geo,
                datum_name=datum,
                units="meters" if is_proj else "degrees",
                utm_zone=utm_zone,
                utm_hemisphere=utm_hemi,
            )
        except Exception:
            return CRSInfo(
                crs_string=crs_str,
                is_projected=True,
                is_geographic=False,
                units="meters",
            )
