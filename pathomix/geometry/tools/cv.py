from shapely.geometry import Polygon
import cv2

def get_shapely_poly(
    contours, hierarchy,  process_hierarchy=True
):
    """
    Convert contours into Shapely polygons, optionally processing the hierarchy to handle nested structures.

    Args:
        contours: List of contours, each represented as an array of points.
        hierarchy: Contour hierarchy array containing relationships between contours.
        scale_factor: Scaling factor to adjust the contour coordinates.
        shift_x: Horizontal shift to apply to contour coordinates.
        shift_y: Vertical shift to apply to contour coordinates.
        process_hierarchy: Whether to process contour relationships based on hierarchy.

    Returns:
        List of Shapely `Polygon` objects.
    """

    assert len(contours) > 0, "No contours to process"
    polys = []

    if process_hierarchy:
        idx_map = get_idx_map(contours, hierarchy)

        for outer_idx, inner_idxs in idx_map.items():
            outer_contour = contours[outer_idx]
            if outer_contour.shape[0] < 4:
                continue
            outer_contour = [((point[0][0]),(point[0][1]),)for point in outer_contour]

            holes = []
            if len(inner_idxs) > 0:
                for inner_idx in inner_idxs:
                    inner_contour = contours[inner_idx]
                    if inner_contour.shape[0] < 4:
                        continue
                    inner_contour = [((point[0][0]),(point[0][1]),)for point in inner_contour]
                    holes.append(inner_contour)

            poly = Polygon(shell=outer_contour, holes=holes)
            polys.append(poly)
    else:
        for contour in contours:
            if contour.shape[0] < 4:
                continue
            contour = [((point[0][0]),(point[0][1]),) for point in contour]
            poly = Polygon(contour)
            polys.append(poly)

    return polys


def get_idx_map(contours, hierarchy):
    """
    Create a mapping of contour indices based on their hierarchical relationships.

    Args:
        contours: List of contours represented as arrays of points.
        hierarchy: Contour hierarchy array containing relationships.

    Returns:
        Dictionary mapping parent contour indices to lists of child contour indices.
    """
    contour_status = _get_contour_status(contours, hierarchy)
    idx_map = {}
    idx_map = dict.fromkeys(contour_status["solo"], [])
    idx_map = _get_hierarchy_idx_map(
        idx_map, contour_status["only_daughter"], hierarchy
    )
    idx_map = _get_hierarchy_idx_map(
        idx_map, contour_status["parent_daughter"], hierarchy
    )

    return idx_map

def _get_contour_status(contours, hierarchy):
    """
    Classify contours based on their relationships in the hierarchy.

    Args:
        contours: List of contours represented as arrays of points.
        hierarchy: Contour hierarchy array containing relationships.

    Returns:
        Dictionary with keys:
            - 'solo': Contours with no parent or child.
            - 'only_parent': Contours with no parent but having children.
            - 'only_daughter': Contours with a parent but no children.
            - 'parent_daughter': Contours with both a parent and children.
    """
    solo = []
    only_parent = []
    only_daughter = []
    parent_daughter = []

    # Traverse each contour and classify
    for idx, contour in enumerate(contours):
        h = hierarchy[0][idx]

        parent_idx = h[3]  # Index of the parent contour
        child_idx = h[2]  # Index of the first child contour

        if parent_idx == -1 and child_idx == -1:
            solo.append(idx)
        elif parent_idx == -1 and child_idx != -1:
            only_parent.append(idx)
        elif parent_idx != -1 and child_idx == -1:
            only_daughter.append(idx)
        elif parent_idx != -1 and child_idx != -1:
            parent_daughter.append(idx)

    assert len(solo) + len(only_parent) + len(only_daughter) + len(
        parent_daughter
    ) == len(contours)

    contour_status = {}
    contour_status["solo"] = solo
    contour_status["only_parent"] = only_parent
    contour_status["only_daughter"] = only_daughter
    contour_status["parent_daughter"] = parent_daughter

    return contour_status


def _get_hierarchy_idx_map(idx_map, contour_status, hierarchy):
    """
    Update the contour index map by processing a specific set of contour statuses.

    Args:
        idx_map: Existing mapping of parent indices to child indices.
        contour_status: Indices of contours to be processed.
        hierarchy: Contour hierarchy array containing relationships.

    Returns:
        Updated mapping of parent indices to child indices.
    """
    for idx in contour_status:
        h = hierarchy[0][idx]

        parent_idx = h[3]  # Index of the parent contour
        child_idx = h[2]  # Index of the first child contour

        found = False
        while not found:
            parent = hierarchy[0][parent_idx]
            if parent[-1] == -1:
                found = True
            else:
                parent_idx = parent[-1]

        if parent_idx in idx_map:
            idx_map[parent_idx].append(idx)
        else:
            idx_map[parent_idx] = []
            idx_map[parent_idx].append(idx)

    return idx_map
