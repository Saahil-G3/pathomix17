import cv2
import geojson
import warnings
import numpy as np
from shapely.affinity import scale
from shapely.geometry import mapping
from shapely.geometry import box as Box


IS_COLLECTION = {}
IS_COLLECTION["Point"] = False
IS_COLLECTION["Polygon"] = False
IS_COLLECTION["LineString"] = False
IS_COLLECTION["LinearRing"] = False
IS_COLLECTION["MultiPoint"] = True
IS_COLLECTION["MultiPolygon"] = True
IS_COLLECTION["MultiLineString"] = True
IS_COLLECTION["GeometryCollection"] = True

def geom_to_geojson(geom):
    """
    Converts a Shapely geometry object to a GeoJSON feature.

    Args:
        geom (shapely.geometry.base.BaseGeometry): A Shapely geometry object.

    Returns:
        geojson.Feature: A GeoJSON feature representation of the input geometry.
    """
    geojson_feature = geojson.Feature(geometry=mapping(geom))
    return geojson_feature

def get_box(x, y, height, width):
    """
    Creates a rectangular bounding box geometry.

    Args:
        x (float): The x-coordinate of the box's lower-left corner.
        y (float): The y-coordinate of the box's lower-left corner.
        height (float): The height of the box.
        width (float): The width of the box.

    Returns:
        shapely.geometry.Polygon: A rectangular bounding box as a Shapely Polygon.
    """
    return Box(x, y, x + height, y + width)

def flatten_geom_collection(geom):
    """
    Flattens a geometry collection into a dictionary of geometry types.

    Args:
        geom (shapely.geometry.base.BaseGeometry): A Shapely GeometryCollection or any geometry.

    Returns:
        dict: A dictionary where keys are geometry types (e.g., 'Point', 'Polygon') and values are lists of geometries of that type.
    """
    geom_dict = {}
    stack = [geom]

    while stack:
        current_geom = stack.pop()
        geom_type = current_geom.geom_type

        if IS_COLLECTION.get(geom_type, False):
            stack.extend(current_geom.geoms)
        else:
            geom_dict.setdefault(geom_type, []).append(current_geom)

    return geom_dict

def get_numpy_mask_for_geom(geom, scale_factor=1, mask_dims = None, origin =(0,0)):
    if mask_dims is None:
        minx, miny, maxx, maxy = geom.bounds
        width = int((maxx - minx)*scale_factor)
        height = int((maxy - miny)*scale_factor)
        mask_dims = (height, width)
        origin = (minx, miny)

    geom = scale(geom=geom, xfact=scale_factor, yfact=scale_factor, origin=origin)
    geom_dict = flatten_geom_collection(geom)

    if len(geom_dict) > 1:
        warnings.warn(
            f"Multiple geometries detected in tissue mask. Check: {', '.join(geom_dict.keys())}"
        )

    exterior, holes = [], []
    for polygon in geom_dict["Polygon"]:
        polygon_coordinates = get_polygon_coordinates(
            polygon, origin=origin
        )
        exterior.extend(polygon_coordinates[0])
        holes.extend(polygon_coordinates[1])
    
    mask = np.zeros(mask_dims, dtype=np.uint8)
    cv2.fillPoly(mask, exterior, 1)
    if len(holes) > 0:
        cv2.fillPoly(mask, holes, 0)

    return mask

def get_polygon_coordinates(polygon, origin=None):
    
    if origin is None:
        origin = np.zeros(2, dtype=np.float32)

    # Exterior coordinates
    exterior = np.array(polygon.exterior.coords, dtype=np.float32)
    np.subtract(exterior, origin, out=exterior)  # In-place origin adjustment
    #np.multiply(exterior, scale_factor, out=exterior)  # In-place scaling
    exterior = np.round(exterior).astype(np.int32)

    # Interior coordinates (holes)
    holes = [
        np.round(
            (np.array(interior.coords, dtype=np.float32) - origin) #* scale_factor
        ).astype(np.int32)
        for interior in polygon.interiors
    ]

    return [exterior], holes