import ezdxf


def get_orientation(obj):
    return 'H' if obj['width'] / obj['height'] > 1.5 else 'V'


def center_dist(a, b):
    return ((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2) ** 0.5


def export_to_dxf(final_results, image_shape, output_path='rzut.dxf'):
    print("\n--- Eksport DXF ---")

    walls = [o for o in final_results if o['class'] == 'wall']
    dim_values = [o for o in final_results if o['class'] == 'dimension_value' and 'value' in o]

    print(f"Znaleziono: {len(walls)} ścian, {len(dim_values)} wartości wymiarów")

    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    SNAP_DIST = max(200, int(max(image_shape[:2]) * 0.15))
    print(f"SNAP_DIST = {SNAP_DIST}px (obraz: {image_shape[1]}x{image_shape[0]})")

    v_walls_list = []
    h_walls_list = []
    for w_idx, w in enumerate(walls):
        orient = get_orientation(w)
        entry = {
            'w_idx': w_idx, 'px_x': w['x'], 'px_y': w['y'],
            'half_w': w['width'] / 2, 'half_h': w['height'] / 2,
        }
        if orient == 'V':
            v_walls_list.append(entry)
        else:
            h_walls_list.append(entry)

    v_walls_list.sort(key=lambda w: w['px_x'])
    h_walls_list.sort(key=lambda w: w['px_y'])
    n_v = len(v_walls_list)
    n_h = len(h_walls_list)

    v_labels = [f"x={v['px_x']:.0f}" for v in v_walls_list]
    h_labels = [f"y={h['px_y']:.0f}" for h in h_walls_list]
    print(f"\nŚciany V (lewa→prawa): {v_labels}")
    print(f"Ściany H (góra→dół):   {h_labels}")

    def find_v_at_h_endpoint(hw, endpoint):
        target_x = hw['px_x'] - hw['half_w'] if endpoint == 'left' else hw['px_x'] + hw['half_w']
        best_vi, best_dist = None, SNAP_DIST
        for vi, vw in enumerate(v_walls_list):
            if not (vw['px_y'] - vw['half_h'] - SNAP_DIST <= hw['px_y'] <= vw['px_y'] + vw['half_h'] + SNAP_DIST):
                continue
            d = abs(vw['px_x'] - target_x)
            if d < best_dist:
                best_dist = d
                best_vi = vi
        return best_vi

    def find_h_at_v_endpoint(vw, endpoint):
        target_y = vw['px_y'] - vw['half_h'] if endpoint == 'top' else vw['px_y'] + vw['half_h']
        best_hi, best_dist = None, SNAP_DIST
        for hi, hw in enumerate(h_walls_list):
            if not (hw['px_x'] - hw['half_w'] - SNAP_DIST <= vw['px_x'] <= hw['px_x'] + hw['half_w'] + SNAP_DIST):
                continue
            d = abs(hw['px_y'] - target_y)
            if d < best_dist:
                best_dist = d
                best_hi = hi
        return best_hi

    h_conn = {}
    v_conn = {}

    for hi, hw in enumerate(h_walls_list):
        h_conn[hi] = (find_v_at_h_endpoint(hw, 'left'), find_v_at_h_endpoint(hw, 'right'))
        lv, rv = h_conn[hi]
        print(f"  H[{hi}] y={hw['px_y']:.0f}: lewy→V[{lv}] prawy→V[{rv}]")

    for vi, vw in enumerate(v_walls_list):
        v_conn[vi] = (find_h_at_v_endpoint(vw, 'top'), find_h_at_v_endpoint(vw, 'bottom'))
        th, bh = v_conn[vi]
        print(f"  V[{vi}] x={vw['px_x']:.0f}: góra→H[{th}] dół→H[{bh}]")

    print("\n--- Identyfikacja luk (gaps) ---")

    h_gaps = []
    for i in range(n_v - 1):
        gap = {
            'left_vi': i, 'right_vi': i + 1,
            'px_left': v_walls_list[i]['px_x'],
            'px_right': v_walls_list[i + 1]['px_x'],
            'px_center': (v_walls_list[i]['px_x'] + v_walls_list[i + 1]['px_x']) / 2,
            'px_span': v_walls_list[i + 1]['px_x'] - v_walls_list[i]['px_x'],
        }
        h_gaps.append(gap)
        print(f"  H gap {i}: V[{i}]↔V[{i+1}] px=[{gap['px_left']:.0f}, {gap['px_right']:.0f}] span={gap['px_span']:.0f}")

    v_gaps = []
    for j in range(n_h - 1):
        gap = {
            'top_hi': j, 'bottom_hi': j + 1,
            'px_top': h_walls_list[j]['px_y'],
            'px_bottom': h_walls_list[j + 1]['px_y'],
            'px_center': (h_walls_list[j]['px_y'] + h_walls_list[j + 1]['px_y']) / 2,
            'px_span': h_walls_list[j + 1]['px_y'] - h_walls_list[j]['px_y'],
        }
        v_gaps.append(gap)
        print(f"  V gap {j}: H[{j}]↔H[{j+1}] px=[{gap['px_top']:.0f}, {gap['px_bottom']:.0f}] span={gap['px_span']:.0f}")

    print("\n--- Dopasowanie wymiarów do luk ---")

    MARGIN = max(image_shape[:2]) * 0.05
    candidates = []
    for dv_idx, dv in enumerate(dim_values):
        dv_x, dv_y = dv['x'], dv['y']

        for g_idx, gap in enumerate(h_gaps):
            if gap['px_left'] - MARGIN <= dv_x <= gap['px_right'] + MARGIN:
                dist = abs(dv_x - gap['px_center'])
                score = dist / max(gap['px_span'], 1)
                candidates.append((score, 'H', g_idx, dv_idx))

        for g_idx, gap in enumerate(v_gaps):
            if gap['px_top'] - MARGIN <= dv_y <= gap['px_bottom'] + MARGIN:
                dist = abs(dv_y - gap['px_center'])
                score = dist / max(gap['px_span'], 1)
                candidates.append((score, 'V', g_idx, dv_idx))

    candidates.sort(key=lambda c: c[0])
    h_gap_values = [None] * len(h_gaps)
    v_gap_values = [None] * len(v_gaps)
    used_dvs = set()
    used_h_gaps = set()
    used_v_gaps = set()

    for score, gap_type, g_idx, dv_idx in candidates:
        if dv_idx in used_dvs:
            continue
        if gap_type == 'H' and g_idx in used_h_gaps:
            continue
        if gap_type == 'V' and g_idx in used_v_gaps:
            continue

        val = dim_values[dv_idx]['value']
        if gap_type == 'H':
            h_gap_values[g_idx] = val
            used_h_gaps.add(g_idx)
            print(f"  dim_value={val} → H gap {g_idx} (V[{h_gaps[g_idx]['left_vi']}]↔V[{h_gaps[g_idx]['right_vi']}]) score={score:.3f}")
        else:
            v_gap_values[g_idx] = val
            used_v_gaps.add(g_idx)
            print(f"  dim_value={val} → V gap {g_idx} (H[{v_gaps[g_idx]['top_hi']}]↔H[{v_gaps[g_idx]['bottom_hi']}]) score={score:.3f}")
        used_dvs.add(dv_idx)

    for dv_idx, dv in enumerate(dim_values):
        if dv_idx not in used_dvs:
            print(f"  Nieprzypisany dim_value={dv['value']} @ ({dv['x']:.0f}, {dv['y']:.0f})")

    print(f"H gap values: {h_gap_values}")
    print(f"V gap values: {v_gap_values}")

    known_h_scales = []
    for i, gap in enumerate(h_gaps):
        if h_gap_values[i] is not None and gap['px_span'] > 0:
            known_h_scales.append(h_gap_values[i] / gap['px_span'])
    if known_h_scales:
        avg_h_scale = sum(known_h_scales) / len(known_h_scales)
        for i in range(len(h_gaps)):
            if h_gap_values[i] is None:
                h_gap_values[i] = h_gaps[i]['px_span'] * avg_h_scale
                print(f"  Pixel fallback: H gap {i} = {h_gap_values[i]:.1f} (scale={avg_h_scale:.4f})")

    known_v_scales = []
    for j, gap in enumerate(v_gaps):
        if v_gap_values[j] is not None and gap['px_span'] > 0:
            known_v_scales.append(v_gap_values[j] / gap['px_span'])
    if known_v_scales:
        avg_v_scale = sum(known_v_scales) / len(known_v_scales)
        for j in range(len(v_gaps)):
            if v_gap_values[j] is None:
                v_gap_values[j] = v_gaps[j]['px_span'] * avg_v_scale
                print(f"  Pixel fallback: V gap {j} = {v_gap_values[j]:.1f} (scale={avg_v_scale:.4f})")

    print("\n--- Pozycje DXF (sekwencyjne) ---")

    v_x = [None] * n_v
    v_x[0] = 0.0
    for i in range(len(h_gaps)):
        if v_x[i] is not None and h_gap_values[i] is not None:
            v_x[i + 1] = v_x[i] + h_gap_values[i]

    h_y = [None] * n_h
    if n_h > 0:
        h_y[n_h - 1] = 0.0
    for j in range(len(v_gaps) - 1, -1, -1):
        if h_y[j + 1] is not None and v_gap_values[j] is not None:
            h_y[j] = h_y[j + 1] + v_gap_values[j]

    print(f"v_x = {v_x}")
    print(f"h_y = {h_y}")

    print("\n--- Rysowanie ścian ---")
    drawn = 0

    for hi, hw in enumerate(h_walls_list):
        y = h_y[hi]
        if y is None:
            print(f"  POMINIĘTO H[{hi}] (y nieznane)")
            continue
        lv, rv = h_conn[hi]

        if lv is not None and v_x[lv] is not None:
            x1 = v_x[lv]
        else:
            print(f"  POMINIĘTO H[{hi}] (brak lewej V wall z pozycją)")
            continue

        if rv is not None and v_x[rv] is not None:
            x2 = v_x[rv]
        else:
            print(f"  POMINIĘTO H[{hi}] (brak prawej V wall z pozycją)")
            continue

        msp.add_line((x1, y), (x2, y))
        print(f"  DXF H[{hi}]: ({x1:.1f}, {y:.1f}) → ({x2:.1f}, {y:.1f})  [długość={abs(x2-x1):.1f}]")
        drawn += 1

    for vi, vw in enumerate(v_walls_list):
        x = v_x[vi]
        if x is None:
            print(f"  POMINIĘTO V[{vi}] (x nieznane)")
            continue
        th, bh = v_conn[vi]

        if bh is not None and h_y[bh] is not None:
            y1 = h_y[bh]
        else:
            print(f"  POMINIĘTO V[{vi}] (brak dolnej H wall z pozycją)")
            continue

        if th is not None and h_y[th] is not None:
            y2 = h_y[th]
        else:
            print(f"  POMINIĘTO V[{vi}] (brak górnej H wall z pozycją)")
            continue

        msp.add_line((x, y1), (x, y2))
        print(f"  DXF V[{vi}]: ({x:.1f}, {y1:.1f}) → ({x:.1f}, {y2:.1f})  [długość={abs(y2-y1):.1f}]")
        drawn += 1

    unassigned_dvs = [dv for dv_idx, dv in enumerate(dim_values) if dv_idx not in used_dvs]

    if unassigned_dvs:
        print(f"\n--- Domknięcie: {len(unassigned_dvs)} nieprzypisanych wymiarów ---")

        open_endpoints = []

        for vi, vw in enumerate(v_walls_list):
            x = v_x[vi]
            if x is None:
                continue
            th, bh = v_conn[vi]
            has_top = (th is not None and h_y[th] is not None)
            has_bottom = (bh is not None and h_y[bh] is not None)

            if has_top == has_bottom:
                continue

            best_dv, best_dist = None, float('inf')
            for dv in unassigned_dvs:
                d = ((vw['px_x'] - dv['x'])**2 + (vw['px_y'] - dv['y'])**2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_dv = dv
            if best_dv is None:
                continue
            val = best_dv['value']

            if has_bottom and not has_top:
                top_y = h_y[bh] + val
                open_endpoints.append({'x': x, 'y': top_y, 'type': 'V', 'idx': vi, 'side': 'top'})
                print(f"  V[{vi}] otwarta u góry: ({x:.1f}, {top_y:.1f}) [użyto dim={val}]")
            elif has_top and not has_bottom:
                bot_y = h_y[th] - val
                open_endpoints.append({'x': x, 'y': bot_y, 'type': 'V', 'idx': vi, 'side': 'bottom'})
                print(f"  V[{vi}] otwarta u dołu: ({x:.1f}, {bot_y:.1f}) [użyto dim={val}]")

        for hi, hw in enumerate(h_walls_list):
            y = h_y[hi]
            if y is None:
                continue
            lv, rv = h_conn[hi]
            has_left = (lv is not None and v_x[lv] is not None)
            has_right = (rv is not None and v_x[rv] is not None)

            if has_left == has_right:
                continue

            best_dv, best_dist = None, float('inf')
            for dv in unassigned_dvs:
                d = ((hw['px_x'] - dv['x'])**2 + (hw['px_y'] - dv['y'])**2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_dv = dv
            if best_dv is None:
                continue
            val = best_dv['value']

            if has_left and not has_right:
                right_x = v_x[lv] + val
                open_endpoints.append({'x': right_x, 'y': y, 'type': 'H', 'idx': hi, 'side': 'right'})
                print(f"  H[{hi}] otwarta z prawej: ({right_x:.1f}, {y:.1f}) [użyto dim={val}]")
            elif has_right and not has_left:
                left_x = v_x[rv] - val
                open_endpoints.append({'x': left_x, 'y': y, 'type': 'H', 'idx': hi, 'side': 'left'})
                print(f"  H[{hi}] otwarta z lewej: ({left_x:.1f}, {y:.1f}) [użyto dim={val}]")

        Y_TOL = 50
        v_open = [ep for ep in open_endpoints if ep['type'] == 'V']
        for i in range(len(v_open)):
            for j in range(i + 1, len(v_open)):
                ep1, ep2 = v_open[i], v_open[j]
                if abs(ep1['y'] - ep2['y']) < Y_TOL:
                    y_conn = (ep1['y'] + ep2['y']) / 2
                    x1_conn = min(ep1['x'], ep2['x'])
                    x2_conn = max(ep1['x'], ep2['x'])
                    length = abs(x2_conn - x1_conn)

                    msp.add_line((x1_conn, y_conn), (x2_conn, y_conn))
                    drawn += 1
                    match = [dv for dv in unassigned_dvs if abs(dv['value'] - length) < length * 0.15]
                    label = f", wymiar={match[0]['value']}" if match else ""
                    print(f"  DOMKNIĘCIE H: ({x1_conn:.1f}, {y_conn:.1f}) → ({x2_conn:.1f}, {y_conn:.1f})  [długość={length:.1f}{label}]")

        h_open = [ep for ep in open_endpoints if ep['type'] == 'H']
        for i in range(len(h_open)):
            for j in range(i + 1, len(h_open)):
                ep1, ep2 = h_open[i], h_open[j]
                if abs(ep1['x'] - ep2['x']) < Y_TOL:
                    x_conn = (ep1['x'] + ep2['x']) / 2
                    y1_conn = min(ep1['y'], ep2['y'])
                    y2_conn = max(ep1['y'], ep2['y'])
                    length = abs(y2_conn - y1_conn)

                    msp.add_line((x_conn, y1_conn), (x_conn, y2_conn))
                    drawn += 1
                    match = [dv for dv in unassigned_dvs if abs(dv['value'] - length) < length * 0.15]
                    label = f", wymiar={match[0]['value']}" if match else ""
                    print(f"  DOMKNIĘCIE V: ({x_conn:.1f}, {y1_conn:.1f}) → ({x_conn:.1f}, {y2_conn:.1f})  [długość={length:.1f}{label}]")

    doc.saveas(output_path)
    print(f"\nZapisano {drawn} ścian do {output_path} (pominięto {n_v + n_h - drawn} odłączonych)")
