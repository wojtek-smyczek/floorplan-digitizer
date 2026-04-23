def nms_dimension_values(predictions, overlap_threshold=0.3):
    """Usuwa nakładające się detekcje dimension_value (Non-Maximum Suppression).
    Używa intersection/min_area zamiast IoU — lepiej łapie częściowe nakładanie."""
    dim_vals = [p for p in predictions if p['class'] == 'dimension_value']
    others = [p for p in predictions if p['class'] != 'dimension_value']

    if len(dim_vals) == 0:
        return predictions

    dim_vals.sort(key=lambda p: p['confidence'], reverse=True)

    kept = []
    for candidate in dim_vals:
        cx, cy, cw, ch = candidate['x'], candidate['y'], candidate['width'], candidate['height']
        cx1, cy1 = cx - cw/2, cy - ch/2
        cx2, cy2 = cx + cw/2, cy + ch/2

        is_duplicate = False
        for selected in kept:
            sx, sy, sw, sh = selected['x'], selected['y'], selected['width'], selected['height']
            sx1, sy1 = sx - sw/2, sy - sh/2
            sx2, sy2 = sx + sw/2, sy + sh/2

            inter_x1 = max(cx1, sx1)
            inter_y1 = max(cy1, sy1)
            inter_x2 = min(cx2, sx2)
            inter_y2 = min(cy2, sy2)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                candidate_area = cw * ch
                selected_area = sw * sh
                min_area = min(candidate_area, selected_area)
                overlap_ratio = inter_area / min_area

                if overlap_ratio > overlap_threshold:
                    is_duplicate = True
                    selected['x'] = (min(cx1, sx1) + max(cx2, sx2)) / 2
                    selected['y'] = (min(cy1, sy1) + max(cy2, sy2)) / 2
                    selected['width'] = max(cx2, sx2) - min(cx1, sx1)
                    selected['height'] = max(cy2, sy2) - min(cy1, sy1)
                    break

        if not is_duplicate:
            kept.append(candidate)

    return kept + others


def merge_adjacent_dimension_values(predictions):
    """Scala sąsiednie detekcje dimension_value na tej samej linii.
    Np. '1' + '100' obok siebie → jedno złączone bbox do ponownego OCR."""
    dim_vals = [p for p in predictions if p['class'] == 'dimension_value']
    others = [p for p in predictions if p['class'] != 'dimension_value']

    if len(dim_vals) < 2:
        return predictions

    merged = []
    used = set()

    for i, a in enumerate(dim_vals):
        if i in used:
            continue
        ax1, ay1 = a['x'] - a['width']/2, a['y'] - a['height']/2
        ax2, ay2 = a['x'] + a['width']/2, a['y'] + a['height']/2

        group = [a]
        used.add(i)

        for j, b in enumerate(dim_vals):
            if j in used:
                continue
            bx1, by1 = b['x'] - b['width']/2, b['y'] - b['height']/2
            bx2, by2 = b['x'] + b['width']/2, b['y'] + b['height']/2

            y_overlap = min(ay2, by2) - max(ay1, by1)
            min_h = min(a['height'], b['height'])
            x_gap = max(bx1 - ax2, ax1 - bx2, 0)
            max_w = max(a['width'], b['width'])

            if y_overlap > min_h * 0.5 and x_gap < max_w * 0.5:
                group.append(b)
                used.add(j)
                ax1 = min(ax1, bx1)
                ay1 = min(ay1, by1)
                ax2 = max(ax2, bx2)
                ay2 = max(ay2, by2)

        if len(group) > 1:
            merged_entry = dict(group[0])
            merged_entry['x'] = (ax1 + ax2) / 2
            merged_entry['y'] = (ay1 + ay2) / 2
            merged_entry['width'] = ax2 - ax1
            merged_entry['height'] = ay2 - ay1
            if 'value' in merged_entry:
                del merged_entry['value']
            merged.append(merged_entry)
            print(f"  Scalono {len(group)} sąsiednich dim_value w jeden bbox")
        else:
            merged.append(a)

    return merged + others
