import os

import pygame
from pygame.locals import *
import math
import itertools

WIDTH = 1024
HEIGHT = 1024
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 1024
GAME_WIDTH = 256.0  # internal game logic resolution
GAME_HEIGHT = 256.0
X_SCALE = (WIDTH / GAME_WIDTH)
Y_SCALE = (HEIGHT / GAME_HEIGHT)

TPS = 60  # number of game updates per second
SKIP_TICKS = 1000 / TPS  # ms to start skipping frames
MAX_FRAMESKIP = 5  # no we calc max updates (if we are behind) before displaying
MAX_TIMESKIP = 2000  # max Time we try to catch up until we just reset counter

MM_GRID = "grid"         # MoveModes
MM_SPEED_FAST = "fast"
MM_SPEED_SLOW = "slow"
MM_VERTICAL = "free_vertical"
MM_HORIZONTAL = "free_horizontal"

BLACK = FONT_NORMAL = 0
WHITE = CENTER_X = FONT_LARGE = 1
DARKGREY = CENTER_Y = FONT_SCORE = 2
YELLOW = CENTER = 3
MIDRED = NO_BLIT = 4
GREY = 5
RED = 6
BLUE = 7
GREEN = 8
DARKRED = 9
CYAN = 10

fonts = None
screen = None
window_surface = None
logo = None
active_live = None
inactive_live = None
color = [(0, 0, 0), (255, 255, 255), (73, 73, 73),        # BLACK,   WHITE,  DARKGREY
         (255, 255, 73), (217, 92, 74), (128, 128, 128),  # YELLOW,  MIDRED, GREY
         (255, 73, 73), (73, 73, 255), (73, 255, 73),     # RED,     BLUE,   GREEN,
         (145, 36, 18), (0, 127, 127)]                    # DARKRED, CYAN
frame_counter = 0  # counts the frames (updating qix not every frame)
current_player = 0
max_player = 2
# ------- per player --------
playfield = [[], []]  # the array of vertex, holding the borders of the game
old_polys = [[], []]
old_poly_colors = [[], []]
player_coords = [[], []]  # an  x/y coordinate
player_lives = [0, 0]  # num lives of both players
scores = [0, 0]
# ---------------------
players_path = []  # the path the player will draw on screen
player_size = 3.0
player_start = [128, 239]
half_frame_rate = False
pressed_keys = []  # sequence of pressed keys and current state
new_playfield = [(16, 39), (16, 239), (240, 239), (240, 39)]
start_player_lives = 3
move_mode = []  # holds state and sub_state(for movement) of the general game
live_coord = (234, 14)
highscore = [(30000, "QIX") for i in range(10)]
fire_slow = fire_fast = up = down = left = right = False


def hal_blt(img, coords):
    screen.blit(img, (coords[0] * X_SCALE, coords[1] * Y_SCALE))


def hal_load_image(fullname, color_key=None):
    try:
        image = pygame.image.load(fullname)
    except pygame.error as message:
        print('Cannot load image:', fullname)
        raise SystemExit(message)
    image = image.convert()
    if color_key is not None:
        if color_key == -1:
            color_key = image.get_at((0, 0))
        image.set_colorkey(color_key, RLEACCEL)
    return image, image.get_rect()


def hal_fill_poly(point_list, arg_color):
    if len(point_list) > 2:
        transformed = [(pnt[0] * X_SCALE, pnt[1] * Y_SCALE) for pnt in point_list]
        pygame.draw.polygon(screen, arg_color, transformed)


def hal_draw_line(point_1, point_2, arg_color):
    point_1 = (point_1[0] * X_SCALE, point_1[1] * Y_SCALE)
    point_2 = (point_2[0] * X_SCALE, point_2[1] * Y_SCALE)
    pygame.draw.line(screen, arg_color, point_1, point_2)


def get_real_time():
    """ Real time of the system in ms after pygame.init"""
    return pygame.time.get_ticks()


def hal_draw_rect(point_1, point_2, arg_color):
    point_1 = (point_1[0] * X_SCALE, point_1[1] * Y_SCALE)
    point_2 = (point_2[0] * X_SCALE, point_2[1] * Y_SCALE)
    pygame.draw.rect(screen, arg_color, (point_1[0], point_1[1], point_2[0] - point_1[0], point_2[1] - point_1[1]))


def print_at(str_text, coords, txt_color=color[YELLOW], center_flags=0, anti_aliasing=1, use_font=FONT_NORMAL):
    if center_flags & 0x04 == 0:
        text = fonts[use_font].render(str_text, anti_aliasing, txt_color)
    else:
        return fonts[use_font].size(str_text)
    xco = coords[0]
    yco = coords[1]
    w = text.get_rect().width
    h = text.get_rect().height
    if center_flags & 0x01 != 0:
        xco = (WIDTH - w) / 2 / X_SCALE
    if center_flags & 0x02 != 0:
        yco = (HEIGHT - h) / 2 / Y_SCALE
    text_pos = Rect(xco, yco, w, h)
    hal_blt(text, text_pos)
    return text_pos[2:]


def vector_add(v1, v2):
    return [v1[0] + v2[0], v1[1] + v2[1]]


def vector_sub(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1]]


def vector_equal(v1, v2, epsilon=0.00001):
    delta = (v1[0] - v2[0])**2 + (v1[1] - v2[1])**2
    return delta < epsilon


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def distance_point_line(pt, l1, l2, sqrt=True):
    """returns distance between point and line segment
    optionally omits calculating square root if only comparison is needed
    :param pt: 1st point
    :param l1: 1st point of line segment
    :param l2: 2nd point of line segment
    :param sqrt: if false returns squared distance
    :return: (squared) distance between points
    """
    a = pt[0] - l1[0]  # var A = x - x1;
    b = pt[1] - l1[1]  # var B = y - y1;
    c = l2[0] - l1[0]  # var C = x2 - x1;
    d = l2[1] - l1[1]  # var D = y2 - y1;
    dot = a * c + b * d
    len_sq = c * c + d * d
    param = -1
    if len_sq != 0:  # in case of 0 length line
        param = float(dot) / len_sq

    if param < 0:
        xx = l1[0]
        yy = l1[1]
    elif param > 1:
        xx = l2[0]
        yy = l2[1]
    else:
        xx = l1[0] + param * c
        yy = l1[1] + param * d

    dx = pt[0] - xx
    dy = pt[1] - yy
    retval = dx * dx + dy * dy
    if sqrt:
        retval = math.sqrt(retval)
    return retval


def intersect_line(p1, p2, p3, p4, strict=False):
    """
    This function will intersect the two lines given by two points each
    boolean flag strict will determine if 2nd point belongs to line
    (so if line (( 0,  0) - (0,100) ) will intersect
    with line   (-50,100)- (50,100) )
    :param p1: 1st point of first line
    :param p2: 2nd point of first line
    :param p3: 1st point of second line
    :param p4: 2nd point of second line
    :param strict: if true excludes 2nd point of each line
    :return: returns point of intersection or
             () if no intersection or
             the two points, if parallel lines overlap
    """
    retval = ()
    t1 = t2 = 2.0
    d1 = (p2[0] - p1[0], p2[1] - p1[1])
    d2 = (p4[0] - p3[0], p4[1] - p3[1])
    det = float(d1[0] * d2[1] - d2[0] * d1[1])
    if det == 0:  # same direction => parallel lines? or same line?
        d3 = (p3[0] - p1[0], p3[1] - p1[1])  # delta between p1 and p3
        d4 = (p4[0] - p2[0], p4[1] - p2[1])  # delta between p2 and p4
        det2 = float(d1[0] * d3[1] - d3[0] * d1[1])  # determinant to check if delta3 is same as delta1
        det3 = float(d2[0] * d4[1] - d4[0] * d2[1])  # determinant to check if delta3 is same as delta1
        if det2 == 0 and det3 == 0:  # same line
            if d1[0] != 0:  # either d1[0] (dx must be >0 or dy >0 or its not a line)
                t1 = (float(p3[0] - p1[0]) / d1[0])  # calc factor on same line
                t2 = (float(p4[0] - p1[0]) / d1[0])
            elif d1[1] != 0:
                t1 = (float(p3[1] - p1[1]) / d1[1])
                t2 = (float(p4[1] - p1[1]) / d1[1])
            elif d2[0] != 0:  # p1 and p2 are same -> swap p1,p2 with p3,p4
                t1 = (float(p1[0] - p3[0]) / d2[0])
                t2 = (float(p2[0] - p3[0]) / d2[0])
            elif d2[1] != 0:
                t1 = (float(p1[1] - p3[1]) / d2[1])
                t2 = (float(p2[1] - p3[1]) / d2[1])
            else:  # p1 and p2 are same AND p3 and P4 are same: return p1 if they are all same
                if p1 == p3:
                    return p1
        else:  # parallel lines do not intersect
            return ()
        # either one of them is in limit[0..1] or they are on different sides..
        if min(t1, t2) <= 1.0 and max(t1, t2) >= 0.0:
            t1n = max(min(t1, t2), 0.0)
            t2n = min(max(t1, t2), 1.0)
            retval = ((p1[0] + t1n * d1[0], p1[1] + t1n * d1[1]),
                      (p1[0] + t2n * d1[0], p1[1] + t2n * d1[1]))
            if retval[0] == retval[1]:
                retval = retval[0]
    else:
        t1 = float(d2[0] * (p1[1] - p3[1]) - d2[1] * (p1[0] - p3[0])) / det
        t2 = float(d1[0] * (p1[1] - p3[1]) - d1[1] * (p1[0] - p3[0])) / det
        if strict:
            if 0.0 <= t1 < 1.0 and 0.0 <= t2 < 1.0:  # point has to be on line segment
                retval = (p3[0] + t2 * d2[0], p3[1] + t2 * d2[1])
        else:
            if 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0:  # point has to be on line segment
                retval = (p3[0] + t2 * d2[0], p3[1] + t2 * d2[1])
    return retval


def draw_list(p_list, arg_color, closed=True):
    upper_limit = len(p_list)
    if not closed:
        upper_limit -= 1
    for index in range(0, upper_limit):
        hal_draw_line(p_list[index],
                      p_list[(index + 1) % len(p_list)], arg_color)


def remove_double_vertex(polygon):
    """
    will remove all double vertexes in a polygon list TODO use faster ALGO
    to avoid problems in intersect_line, if no line, but the same point
    is given double points will always arise if you start a new path on a corner
    """
    removals = []
    old_v = (-1, -1)
    for index in range(len(polygon) - 1, 0, -1):
        if polygon[index][0] == old_v[0] and polygon[index][1] == old_v[1]:
            removals.append(index)
        old_v = polygon[index]
    for index in removals:
        del polygon[index]
    return polygon


def calc_area(polygon):
    retval = 0
    for index in range(0, len(polygon)):
        v1 = polygon[index]
        v2 = polygon[(index + 1) % len(polygon)]
        retval += (v1[0] * v2[1] - v1[1] * v2[0])
    return retval / 2.0


def is_inside(polygon, candidate, outside_point=(), strict=True):
    """
    will determine, if a given candidate point is inside the polygon
    parameters:
        polygon (list of two dimensional points)
        candidate a 2D-Point which is in question to be in or outside of the poly
        outside_point a point guaranteed to be on the outside, if not given,
                     method will calculate one(slower)
        strict controls, if boundary lines belong to the polygon (False) or not (True)
    returns True, if candidate is inside polygon
            False, if candidate is outside of polygon
    """
    on_line = False
    for index in range(0, len(polygon)):
        vertex1 = polygon[index]
        vertex2 = polygon[(index + 1) % len(polygon)]
        intersect = intersect_line(vertex1, vertex2,  # TODO: use Point-line intersection, not line-line..
                                   candidate, candidate, strict=True)
        if len(intersect) > 0:  # intersection was found
            on_line = True
    if on_line:
        return not strict
    if len(outside_point) != 2:  # if outside_point is not given, create one
        max_x = max_y = min_x = min_y = 0  # calc polys bounding box
        for vertex in polygon:
            if vertex[0] > max_x:
                max_x = vertex[0]
            if vertex[0] < min_x:
                min_x = vertex[0]
            if vertex[1] > max_y:
                max_y = vertex[0]
            if vertex[1] < min_y:
                min_y = vertex[0]
        delta = (max_x - min_x, max_y - min_y)  # diagonal of bounding box
        outside_point = (max_x + delta[0], max_y + delta[1])  # move outside
    intersection_count = 0
    for index in range(0, len(polygon)):
        vertex1 = polygon[index]
        vertex2 = polygon[(index + 1) % len(polygon)]
        intersect = intersect_line(vertex1, vertex2,
                                   outside_point, candidate, strict=True)
        if len(intersect) > 0:  # intersection was found
            if isinstance(intersect[0], float):
                # if type(intersect[0]) == type(float(0)):
                intersection_count += 1
    return (intersection_count % 2) == 1


def find_intersect_index(arg_poly, point, candidates=None, close=True):
    """
    find the line segment on which a given point resides
    :param arg_poly: the list of points forming the (counter-circular) axis aligned polygon
    :param point: the point which should be searched
    :param candidates: a part of poly array to avoid traversing all segments
    :param close:  True, if the polygon is closed(like playfield)or
                   False if the polygon is open (like path)
    :return:  the indexes into arg_poly (multiple entries if point is on a vertex)
    """
    retval = []
    upper_limit = len(arg_poly)
    if not close:
        upper_limit -= 1
    if candidates is None:
        candidates = range(0, upper_limit)
    for index_src in candidates:
        index_dst = (index_src + 1) % len(arg_poly)
        xl = min(arg_poly[index_src][0], arg_poly[index_dst][0])
        xh = max(arg_poly[index_src][0], arg_poly[index_dst][0])
        yl = min(arg_poly[index_src][1], arg_poly[index_dst][1])
        yh = max(arg_poly[index_src][1], arg_poly[index_dst][1])
        if xl <= (point[0]) <= xh and yl <= (point[1]) <= yh:
            retval.append(index_src)
    return retval


def split_polygon(arg_poly, path):
    """
     this is one of the heart routines of qix, the splitPolygon method,
     which will be called if a new polygon must be drawn. It returns 2 Polygons:
        the one which should be filled
        the other one which is the new free space(with the qix in it..)
    """
    poly1 = []
    poly2 = []
    # 1st: get line from poly of start point
    start_index = find_intersect_index(arg_poly, path[0])[0]
    # 2nd: get line from poly from end point
    end_index = find_intersect_index(arg_poly, path[-1])[0]
    # construct 2 new polys by splitting the poly on start and end edges and
    # inserting the path in both polys
    if start_index > end_index:
        start_index, end_index = end_index, start_index
        path.reverse()
    # check path orientation by scalar product(?) if starts on same segment
    if start_index == end_index:
        dx_path = path[-1][0] - path[0][0]
        dy_path = path[-1][1] - path[0][1]
        dx_poly = arg_poly[(start_index + 1) % len(arg_poly)][0] - arg_poly[start_index][0]
        dy_poly = arg_poly[(start_index + 1) % len(arg_poly)][1] - arg_poly[start_index][1]
        if dx_path * dx_poly + dy_path * dy_poly < 0:
            start_index, end_index = end_index, start_index
            path.reverse()
    seg1 = arg_poly[0:start_index + 1]
    seg2 = arg_poly[start_index + 1:end_index + 1]
    seg3 = arg_poly[end_index + 1:]

    path1 = list(path)
    poly1.extend(seg1)
    poly1.extend(path1)
    poly1.extend(seg3)

    path2 = list(path)
    path2.reverse()
    poly2.extend(path2)
    poly2.extend(seg2)
    poly1 = remove_double_vertex(poly1)
    poly2 = remove_double_vertex(poly2)
    return poly1, poly2


def get_first_collision(collision, ignore_pt=(-1, -1)):
    """examines a collision result and returns either the ignore_pt, if collision only happened in
    ignore_pt or the other point of the collision"""
    candidate = ignore_pt  # reset candidate to default value and check the collision
    for collide in collision:
        if type(collide[0]) == tuple:  # result is a line-> take vertex which is not ignore_point
            for vertex in collide:
                if not vector_equal(vertex, ignore_pt):
                    candidate = vertex
                    break
        elif type(collide[0]) == float:  # result is a single point; only valid, if not ignore_point
            if not vector_equal(collide, ignore_pt):
                candidate = collide
                break
    return candidate


def check_line_vs_poly(p1, p2, poly_path, close=True, sort=True):
    """collides a line with a polygon
    returns a list of points or lines(collinear) with collisions"""
    retval = []
    for p3, p4 in pairwise(poly_path):  # check each line segment
        res = intersect_line(p1, p2, p3, p4)
        if len(res) > 0:
            retval.append(res)
    if close:
        res = intersect_line(p1, p2, poly_path[-1], poly_path[0])  # don't forget the closing line..
        if len(res) > 0:
            retval.append(res)

    def local_distance(px):
        if type(px[0]) == tuple:
            dist = distance_point_line(p1, px[0], px[1], False)
        else:
            dist = (px[0] - p1[0]) ** 2 + (px[1] - p1[1]) ** 2
        return dist

    if sort:
        retval.sort(key=local_distance)
    return retval


def paint_score():
    print_at("%d  %s" % (highscore[0][0], highscore[0][1]), (0, 13),
             txt_color=color[YELLOW], center_flags=CENTER_X, anti_aliasing=0)
    for index in range(max_player):      # paint score for both player
        dim = print_at(str(scores[index]), (0, 0), color[WHITE], center_flags=NO_BLIT, use_font=FONT_SCORE)
        coords = (232 - dim[0] / X_SCALE, 16 + index * 11)
        print_at(str(scores[index]), coords, color[WHITE], use_font=FONT_SCORE)


def paint_playerpath():
    idx = DARKRED  # paint playerpath
    if move_mode[1] == MM_SPEED_FAST:
        idx = CYAN
    draw_list(players_path, color[idx], False)


def paint_claimed_and_lives():
    print_at("CLAIMED", (0, 22), txt_color=color[YELLOW], center_flags=CENTER_X, anti_aliasing=0)
    print_at("0%  75%", (0, 29), txt_color=color[YELLOW], center_flags=CENTER_X, anti_aliasing=0)
    for player in range(max_player):
        for index in range(start_player_lives):
            start_coord = [(live_coord[0] + (index // 3) * 4),
                           (live_coord[1] + 11 * current_player + (index % 3) * 4)]
            if player_lives[player] == start_player_lives - index:
                hal_blt(active_live, start_coord)
            else:
                hal_blt(inactive_live, start_coord)


def paint_player():
    pos = player_coords[current_player]
    player = [vector_add(pos, (-player_size, 0)), vector_add(pos, (0, player_size)),
              vector_add(pos, (player_size, 0)), vector_add(pos, (0, -player_size))]
    draw_list(player, color[RED], True)
    if X_SCALE > 1.0:
        hal_draw_rect(vector_add(pos, (-1, -1)), vector_add(pos, (1, 1)), color[WHITE])
    else:
        hal_draw_rect(pos, vector_add(pos, (1, 1)), color[WHITE])  # add single pixel (pygame draws a 3x3 rect on w=0)


def paint_playfield():
    for idx, iter_poly in enumerate(old_polys[current_player]):
        hal_fill_poly(iter_poly, old_poly_colors[current_player][idx])
    for iter_poly in old_polys[current_player]:
        draw_list(iter_poly, color[WHITE])
    draw_list(playfield[current_player], color[WHITE])


def paint_game():
    hal_blt(logo, (24, 16))
    paint_score()
    paint_playfield()
    paint_claimed_and_lives()
    paint_playerpath()
    paint_player()


def move_player(movement):
    global move_mode, players_path, half_frame_rate, old_polys, old_poly_colors, scores, playfield
    half_frame_rate = False
    if [0, 0] == movement:
        return player_coords[current_player]
    if move_mode[0] != MM_GRID and move_mode[1] == MM_SPEED_SLOW and frame_counter % 2 == 1:
        half_frame_rate = True
        return player_coords[current_player]  # process slow movement with half of the framerate to avoid floats
    # get all lines our player stands on..(max. 2, if player stays on corner)
    possible_move = []
    if move_mode[0] == MM_GRID:
        possible_move = find_intersect_index(playfield[current_player], player_coords[current_player])
    candidate = list(player_coords[current_player])
    candidate = vector_add(candidate, movement)

    if len(possible_move) > 0:  # Player is standing on grid
        new_result = find_intersect_index(playfield[current_player], candidate)
        # if candidate is still on same segment fall through..
        if len(list(set(possible_move) & set(new_result))) == 0:
            # we are leaving the segment, either by overshooting ... or starting a new line
            # go single pixel into direction of candidate
            short_candidate = list(player_coords[current_player])
            if movement[0] != 0:
                short_candidate[0] += math.copysign(1, movement[0])
            if movement[1] != 0:
                short_candidate[1] += math.copysign(1, movement[1])
            if is_inside(playfield[current_player], short_candidate, strict=True) and (fire_slow or fire_fast):
                # initialize a new PATH
                move_mode[0] = MM_HORIZONTAL
                if movement[0] == 0:
                    move_mode[0] = MM_VERTICAL
                move_mode[1] = MM_SPEED_FAST
                if fire_slow:
                    move_mode[1] = MM_SPEED_SLOW
                collision = check_line_vs_poly(player_coords[current_player], short_candidate,
                                               playfield[current_player], close=True, sort=False)
                start_point = get_first_collision(collision, ignore_pt=player_coords[current_player])
                players_path = [start_point, start_point]
                possible_move = []  # allows to jump into next outer if for free roaming
            else:  # just overshooting..clip it to last corner point
                # intersecting movement with playfield
                tmp = [playfield[current_player][possible_move[0] - 1]]
                for mv in possible_move:
                    tmp.append(playfield[current_player][mv])
                tmp.append(playfield[current_player][(possible_move[-1] + 1) % len(playfield[current_player])])
                collision = check_line_vs_poly(player_coords[current_player], candidate, tmp, close=False)
                candidate = get_first_collision(collision, ignore_pt=player_coords[current_player])

    if len(possible_move) == 0:  # Player is roaming a new line
        if fire_slow or fire_fast:
            old_movemode = move_mode[0]
            if movement[0] != 0:
                move_mode[0] = MM_HORIZONTAL
                movement[1] = 0
            elif movement[1] != 0:
                move_mode[0] = MM_VERTICAL
            if old_movemode != move_mode[0]:  # change of direction -> store coord
                players_path.pop()  # replace last coords as they re now set as cornerstone
                players_path.append((player_coords[current_player][0], player_coords[current_player][1]))
                players_path.append((player_coords[current_player][0], player_coords[current_player][1]))
            else:
                players_path.pop()  # remove last coords as they are current position
                players_path.append((player_coords[current_player][0], player_coords[current_player][1]))
            # check for inter path collision
            collision = check_line_vs_poly(player_coords[current_player], candidate, players_path, close=False)
            candidate_tmp = get_first_collision(collision, ignore_pt=player_coords[current_player])
            if not vector_equal(candidate_tmp, player_coords[current_player]) and \
                    not vector_equal(candidate_tmp, players_path[0]):  # player touches his own path: do not move
                candidate = list(player_coords[current_player])
            players_path.pop()
            players_path.append((candidate[0], candidate[1]))
            # check for playfield collision
            collision = check_line_vs_poly(player_coords[current_player], candidate, playfield[current_player],
                                           close=True)
            candidate_tmp = get_first_collision(collision, ignore_pt=player_coords[current_player])
            if not vector_equal(candidate_tmp, player_coords[current_player]) and \
                    not vector_equal(candidate_tmp, players_path[0]):
                players_path.pop()
                players_path.append((candidate_tmp[0], candidate_tmp[1]))
                players_path = [(int(x[0]), int(x[1])) for x in players_path]
                candidate = players_path[-1]
                # 1st: split playfield
                poly1, poly2 = split_polygon(playfield[current_player], list(players_path))
                old_playfield_area = abs(calc_area(playfield[current_player]))
                # 2nd: check which poly is new playfield(the one with the qix inside)
                playfield[current_player] = poly1
                old_polys[current_player].append(poly2)
                # 3rd: calc points, add color and handle supersparx on path
                i = CYAN
                slow_factor = 1
                if move_mode[1] == MM_SPEED_SLOW:
                    i = DARKRED
                    slow_factor = 2
                playfield_area = abs(calc_area(playfield[current_player]))
                complete = abs(calc_area(new_playfield))
                old_percentage = old_playfield_area * 100 / complete
                new_percentage = playfield_area * 100 / complete
                bonus = int((old_percentage - new_percentage) * 100) * slow_factor
                scores[current_player] += bonus
                old_poly_colors[current_player].append(color[i])
                players_path = []
                move_mode = [MM_GRID, None]
        else:  # in roaming mode only move if fast or slow button is pressed
            candidate = list(player_coords[current_player])
        # we are not allowing moving outside the playfield
    if not is_inside(playfield[current_player], candidate, strict=False):
        candidate = player_coords[current_player]
    return candidate


def handle_movement():
    global move_mode
    movement = [0, 0]
    if fire_fast and move_mode[1] == MM_SPEED_SLOW:
        move_mode[1] = MM_SPEED_FAST
    if left:
        movement[0] -= 1
    if right:
        movement[0] += 1
    if up:
        movement[1] -= 1
    if down:
        movement[1] += 1
    if player_lives[current_player] > 0:
        player_coords[current_player] = move_player(movement)


def reset_playfield(index_player):
    global playfield
    playfield[index_player] = new_playfield


def init():
    global window_surface, screen, logo, fonts, active_live, inactive_live, player_lives, player_coords, move_mode
    pygame.init()
    window_surface = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
    screen = pygame.Surface((WIDTH, HEIGHT))
    fonts = [pygame.font.Font("data/qix-small.ttf", int(8.0 * Y_SCALE)),
             pygame.font.Font("data/qix-large.ttf", int(8.0 * Y_SCALE)),
             pygame.font.Font("data/qix-large.ttf", int(10.0 * Y_SCALE))]
    logo, _ = hal_load_image(os.path.join('data', 'qix_logo.png'))
    logo = pygame.transform.scale(logo, (int(56.0 * X_SCALE), int(20 * Y_SCALE)))
    active_live, _ = hal_load_image(os.path.join('data', 'qix_live_w.png'))
    active_live = pygame.transform.scale(active_live, (int(3.0 * X_SCALE), int(3.0 * Y_SCALE)))
    inactive_live, _ = hal_load_image(os.path.join('data', 'qix_live_r.png'))
    inactive_live = pygame.transform.scale(inactive_live,  (int(3.0 * X_SCALE), int(3.0 * Y_SCALE)))
    reset_playfield(0)
    player_lives = [start_player_lives, start_player_lives]
    player_coords = [player_start, player_start]
    move_mode = [MM_GRID, MM_SPEED_SLOW]


def press_key(key):
    global pressed_keys, left, right, up, down, fire_slow, fire_fast
    if key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
        pressed_keys.append(key)
    if key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
        left = right = up = down = False
    if key == pygame.K_LEFT and pygame.K_RIGHT not in pressed_keys:
        left = True
    if key == pygame.K_RIGHT and pygame.K_LEFT not in pressed_keys:
        right = True
    if key == pygame.K_UP and pygame.K_DOWN not in pressed_keys:
        up = True
    if key == pygame.K_DOWN and pygame.K_UP not in pressed_keys:
        down = True
    if key == pygame.K_LALT:
        fire_slow = True
    if key == pygame.K_LCTRL:
        fire_fast = True


def release_key(key):
    global pressed_keys, left, right, up, down, fire_slow, fire_fast
    if key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
        pressed_keys.remove(key)
    status = False
    if key == pygame.K_LALT:
        fire_slow = status
    if key == pygame.K_LCTRL:
        fire_fast = status
    if len(pressed_keys) > 0:
        status = True
        key = pressed_keys[-1]
    left = right = up = down = False
    if key == pygame.K_LEFT and pygame.K_RIGHT not in pressed_keys:
        left = status
    if key == pygame.K_RIGHT and pygame.K_LEFT not in pressed_keys:
        right = status
    if key == pygame.K_UP and pygame.K_DOWN not in pressed_keys:
        up = status
    if key == pygame.K_DOWN and pygame.K_UP not in pressed_keys:
        down = status


def gameloop():  # https://dewitters.com/dewitters-gameloop/
    global frame_counter
    next_game_tick = get_real_time() - 1
    while 1:
        loops = 0
        while get_real_time() > next_game_tick and loops < MAX_FRAMESKIP:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                if event.type == KEYDOWN:
                    press_key(event.key)
                if event.type == KEYUP:
                    release_key(event.key)
            handle_movement()
            frame_counter += 1
            next_game_tick += SKIP_TICKS
            if get_real_time() > next_game_tick + MAX_TIMESKIP:
                next_game_tick = get_real_time() + SKIP_TICKS
            loops += 1
        screen.fill(color[BLACK])
        paint_game()
        window_tmp = pygame.transform.scale(screen, (WINDOW_WIDTH, WINDOW_HEIGHT))
        window_surface.blit(window_tmp, (0, 0))
        pygame.display.flip()


if __name__ == '__main__':
    print("pyQix")
    print("-----")
    print("(c) 2021 YogaSurfTech https://github.com/YogaSurfTech/pyqix")
    print("A faithful remake of the Taito classic arcade game QIX in Python. ")
    print("On the occasion of the 40th anniversary of the release at 18th. October 2021")
    print("Controls:")
    print("The standard MAME keyset is used:")
    print("<CTRL> is FAST button")
    print("<ALT> is SLOW button")
    print("Cursor keys are Joystick")
    init()
    gameloop()
