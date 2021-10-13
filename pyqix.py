import os
import json

import pygame
from pygame.locals import *
import math
import random
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
GM_GAME = "game"
GM_GAMEOVER = "game_over"
GM_HIGHSCORE = "highscore"
GM_HIGHSCORE_ENTRY = "highscore_entry"

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
game_over = None
active_live = None
inactive_live = None
color = [(0, 0, 0), (255, 255, 255), (73, 73, 73),        # BLACK,   WHITE,  DARKGREY
         (255, 255, 73), (217, 92, 74), (128, 128, 128),  # YELLOW,  MIDRED, GREY
         (255, 73, 73), (73, 73, 255), (73, 255, 73),     # RED,     BLUE,   GREEN,
         (145, 36, 18), (0, 127, 127)]                    # DARKRED, CYAN
frame_counter = 0  # counts the frames (updating qix not every frame)
current_player = 0
max_player = 1
# ------- per player --------
playfield = [[], []]  # the array of vertex, holding the borders of the game
old_polys = [[], []]
old_poly_colors = [[], []]
player_coords = [[], []]  # an  x/y coordinate
player_lives = [0, 0]  # num lives of both players
scores = [0, 0]
qix_coords = [[[], []], [[], []]]  # x/y, x/y coordinates  of qix lines for each player
max_qix = [1, 1]  # number of qixes (in later levels are 2 if you manage to split them you win the level)
# ---------------------
players_path = []  # the path the player will draw on screen
player_size = 3.0
player_start = [128, 239]
half_frame_rate = False
new_playfield = [(16, 39), (16, 239), (240, 239), (240, 39)]
qix_target = [(0, 0), (0, 0)]
qix_speed = [[], []]   # x/y x/y , d1/d2, speed and duration(until direction change) of qix lines
qix_change_counter = 0
qix_min_change = 3
qix_max_change = 15
qix_min_speed = 1
qix_max_speed = 10
qix_color_index = [0, 0]
fuse_sleep = 1000  # time of players wait until fuse starts chasing the player in ms
fuse = [0, 0, 0, False]  # fuse hunts player, if he draws a line and stops[x,y,sleep_counter,visible]
sprt_fuse = []  # the array of the fuse sprites
sparx_respawn = 0  # counts up 37 seconds until next pair of spark is respawned
sparx_super_spawn = 1  # number of sparx spawned until they mutate to supersparx
sparx_super_counter = sparx_super_spawn  # counts down how many sparx are spawned until they mutate to supersparx
all_sparx = []  # sparx:x, y, speed, supersparx(=1, normal=0), IsIOnPlayerPath, PointToStopFlip, PolyToWanderFlip
sprt_sparx = []
sprt_supersparx = []
pressed_keys = []  # sequence of pressed keys and current state
dead_counter = 0
dead_count_dir = 1  # direction of dead anim
is_dead = False
point_of_death = (0, 0)
killed_by_qix = False
dead_bubbles = []   # the three death bubbles: each on e has center, rad, outline, [list of black lines erasing circles]
bubble_radius = [2, 4]
dead_range = 16
dead_anim_segments = 12  # death animation segments: 6 segments for showing 3 bubbles and rest for pause and resolving
dead_anim_pause = 2  # pause between showing bubbles and resolving them
deathray_distance = 4  # distance between diagonal lines
no_of_deathrays = 8  # number of lines
deathray_prolong = .5  # how many pixels every line grows after each step
start_player_lives = 3
game_mode = GM_GAMEOVER
move_mode = []  # holds state and sub_state(for movement) of the general game
game_over_coord = (81, 98)
live_coord = (234, 14)
split_poly = None
highscore_file = os.path.join("data", "highscore.dat")
highscore = [(30000, "QIX") for i in range(10)]
new_highscore_entry = "..."
entry_index = 0
entry_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ 012345789<."
entry_accumulator = 0
trigger_up = trigger_down = trigger_fast = fire_slow = fire_fast = up = down = left = right = False
credit = 0


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


def hal_draw_circle(arg_color, center, radius, fill=1):
    fill = int(min(fill, min(radius * X_SCALE, radius * Y_SCALE)))
    rect = (((center[0] - radius) * X_SCALE), int((center[1] - radius) * Y_SCALE),
            radius * 2 * X_SCALE, radius * 2 * Y_SCALE)
    pygame.draw.ellipse(screen, arg_color, rect, fill)


def get_real_time():
    """ Real time of the system in ms after pygame.init"""
    return pygame.time.get_ticks()


def get_time():
    """in-game time calculated calculated by 1/60th sec tics as updates in game logic"""
    return frame_counter * SKIP_TICKS


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


def set_game_mode(mode):
    global game_mode
    game_mode = mode
    paint_game.wait_counter = -1


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def get_random_vector(arg_max, arg_min=0.0):  # TODO :optimize: rnd_x and get y by pythagoras; length is always 1.0 then
    retval = [random.random() - .5, random.random() - .5]
    length = math.sqrt(retval[0] * retval[0] + retval[1] * retval[1])
    factor = random.random() * (arg_max - arg_min) + arg_min
    retval[0] = (retval[0] / length) * factor
    retval[1] = (retval[1] / length) * factor
    return retval


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


def cut_path(arg_poly, arg_start, arg_end, direction=1):
    """returns shortest path from start to end laying on arg poly
    Condition: start and end are on poly
    :param arg_poly: the list of points forming the axis aligned polygon
    :param arg_start: start point of the new path
    :param arg_end:   end point of the new path
    :param direction:   direction how to count through the poly (value will be clamped to +1/-1)
    :return:  list of points forming the axis aligned path starting with point start and ending with point end
    """
    retval = []
    start_index = find_intersect_index(arg_poly, arg_start, close=True)
    end_index = find_intersect_index(arg_poly, arg_end, close=True)
    swapped = False
    if len(start_index) > 0 and len(end_index) > 0:
        start_index = start_index[0]
        end_index = end_index[0]
        if direction >= 0:
            direction = 1
            while end_index < start_index:
                end_index += len(arg_poly)
            start_index += 1
            end_index += 1
        if direction < 0:
            direction = -1
            while end_index > start_index:
                end_index -= len(arg_poly)
        retval.append(arg_start)
        for index in range(start_index, end_index, direction):
            retval.append(arg_poly[index % len(arg_poly)])
        retval.append(arg_end)
        if swapped:
            retval.reverse()
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


def calc_1d_path(poly_path, close=True):
    """calculates the total length of the path"""
    retval = 0
    for p1, p2 in pairwise(poly_path):  # calc deltas and sum up
        retval += abs(p1[0] - p2[0])
        retval += abs(p1[1] - p2[1])
    if close:
        retval += abs(poly_path[-1][0] - poly_path[0][0])
        retval += abs(poly_path[-1][1] - poly_path[0][1])
    return retval


def calc_vertex_from_1d_path(poly_path, start_vertex, offset, close=True):
    """returns the dot according to (circular) 1d path + offset
       DONE: on overshooting after the end path is considered circular
        :param poly_path the polygon or path in question
        :param start_vertex vertex on poly_path to start
        :param offset distance to new vertex
        :param close poly_path is either circular(True) or a polyline(False)
    """
    retval = list(start_vertex)
    index = find_intersect_index(poly_path, start_vertex, close=close)
    if len(index):
        direction = int(math.copysign(1, offset))
        index = index[-1]
        p1 = start_vertex
        if offset < 0:
            index = (index + 1) % len(poly_path)
            offset = - offset
        while offset > 0:
            index2 = (index + direction) % len(poly_path)  # this wraps around even if close = false!
            p2 = poly_path[index2]
            dx, dy = vector_sub(p2, p1)
            subtract = (abs(dx)) + (abs(dy))
            if subtract < offset:       # segments length smaller than offset ?
                offset -= subtract      # subtract and continue on next segment
                index = index2
                p1 = poly_path[index]
                if not close and index2 == len(poly_path)-1:  # not closed and reached last line segment ?
                    retval[0] = p2[0]             # CLAMP the value to last vertex
                    retval[1] = p2[1]
                    offset = 0                    # exit while loop
            else:
                fx = min(dx, 1)
                if dx < 0:
                    fx = -1
                fy = min(dy, 1)
                if dy < 0:
                    fy = -1
                retval[0] = p1[0] + (offset * fx)
                retval[1] = p1[1] + (offset * fy)
                offset = 0              # exit while loop
    return retval


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


def reset_sparx(index_player, player_pos=None):
    global all_sparx, sparx_respawn, sparx_super_counter
    if player_pos is None:
        player_pos = player_coords[index_player]
    sparx_respawn = 0
    sparx_super_counter = sparx_super_spawn
    x1, y1 = calc_vertex_from_1d_path(playfield[index_player], player_pos,
                                      int(calc_1d_path(playfield[index_player]) / 2))
    all_sparx = [[x1, y1, -1, 0, False, [], []],
                 [x1, y1,  1, 0, False, [], []]]


def tick_sparc_respawn():
    global all_sparx, sparx_respawn, sparx_super_counter, all_sparx
    sparx_respawn += 1
    if sparx_respawn > 37:
        sparx_respawn = 0
        sparx_super_counter -= 1
        tmp_playfield = new_playfield
        coords = ((tmp_playfield[0][0] + tmp_playfield[2][0]) / 2, tmp_playfield[0][1])
        spark_left = [coords[0], coords[1], all_sparx[-2][2], 0, False, [], []]
        spark_right = [coords[0], coords[1], all_sparx[-1][2], 0, False, [], []]
        all_sparx.append(spark_left)
        all_sparx.append(spark_right)
        if sparx_super_counter < 0:  # this will be repeated until player dies or level is completed
            for sparc in all_sparx:
                sparc[3] = 1


def qix_change_color(index):
    global qix_color_index
    qix_colors = [MIDRED, BLUE, GREEN]
    if random.random() * 100 < 7:
        qix_color_index[index] = (qix_color_index[index] + 1) % len(qix_colors)
    return color[qix_colors[qix_color_index[index]]]


def calc_max_exploding_line_steps():
    """  Calculates the number of steps for death_anim to play so that all lines left the screen
    :return: Number of steps so that no line segment is visible on screen anymore (to end dead animation)
    """
    max_steps = (max(  # what is the max number of steps to reach the border of screen
        min(player_coords[current_player][0], player_coords[current_player][1]),
        min(player_coords[current_player][1], GAME_WIDTH - player_coords[current_player][0]),
        min(GAME_HEIGHT - player_coords[current_player][1], GAME_WIDTH - player_coords[current_player][0]),
        min(GAME_HEIGHT - player_coords[current_player][1], player_coords[current_player][0]))
                 / deathray_distance) + no_of_deathrays
    max_steps += max_steps * float(deathray_prolong) / deathray_distance  # adapt the no of jmps for diagonal line segm
    return SKIP_TICKS * 3.0 * max_steps


def death_anim():
    """shows the player animation in dependency of dead_counter (rays of diagonal lines)
       is calculating the number of max dead_counter (to make all lines disappear)
    """
    global dead_counter, killed_by_qix
    max_count = calc_max_exploding_line_steps()
    i_end = int(float(dead_counter)/SKIP_TICKS)
    if killed_by_qix:
        p1 = vector_add(point_of_death, (-1, -1))
        p2 = vector_add(point_of_death, (1, 1))
        hal_draw_rect(p1, p2, color[WHITE])
        bubble_count = 0
        anim_phase = min((dead_counter / float(max_count)), 1.0) * dead_anim_segments
        for bubble in dead_bubbles:
            bubble_count += 1
            if anim_phase > bubble_count + 1:
                hal_draw_circle((255, 255, 255), bubble[0], bubble[1], bubble[2])
        if anim_phase > 6 + dead_anim_pause:
            draw_lines = int(((anim_phase - (6 + dead_anim_pause)) / (dead_anim_segments - 6 - dead_anim_pause))
                             * len(dead_bubbles[0][3]))
            for index in range(0, draw_lines):
                for b in range(0, 3):
                    hal_draw_line(dead_bubbles[2 * b][3][index][0], dead_bubbles[2 * b][3][index][1], (0, 0, 0))
    if i_end < 10:
        i_start = 1
    else:
        i_start = i_end - no_of_deathrays
    for index in range(i_start, i_end):
        for d1, d2 in [(1, 1), (-1, -1), (-1, 1), (1, -1)]:  # d1/d2 are x/y-directions
            pt_end = (player_coords[current_player][0] + d1 * index * deathray_distance,
                      player_coords[current_player][1] + d2 * index * deathray_distance)
            p1 = (pt_end[0] - deathray_prolong * d1 * index, pt_end[1] + deathray_prolong * d2 * index)
            p2 = (pt_end[0] + deathray_prolong * d1 * index, pt_end[1] - deathray_prolong * d2 * index)
            hal_draw_line(p1, p2, (255, 255, 255))
    if dead_counter > max_count:  # all lines out of the screen?
        revive_player()
        killed_by_qix = False
        dead_counter = calc_max_exploding_line_steps()


def paint_highscore():
    print_at("qix KICKERS", (84, 60), color[WHITE], center_flags=CENTER_X, anti_aliasing=0, use_font=FONT_LARGE)
    print_at("SCORE", (59, 90), color[WHITE], center_flags=0x00, anti_aliasing=0, use_font=FONT_LARGE)
    print_at("INITIALS", (147, 90), color[WHITE], center_flags=0x00, anti_aliasing=0, use_font=FONT_LARGE)
    for index in range(10):
        pts, initials = highscore[index]
        dim = print_at(str(pts), (86, (109 + index * 7)), color[YELLOW], center_flags=NO_BLIT, anti_aliasing=0)
        print_at(str(pts), (100-dim[0]/X_SCALE, (109 + index * 7)), color[YELLOW], center_flags=0x00, anti_aliasing=0)
        print_at(str(initials), (168, (109 + index * 7)), color[YELLOW], center_flags=0x00, anti_aliasing=0)


def paint_highscore_entry():
    paint_highscore()
    print_at("SIGN UP", (104, 72))
    print_at("MOVE JOYSTICK UP OR DOWN", (48, 216))
    print_at("PRESS FAST TO ENTER", (66, 226))
    print_at(new_highscore_entry, (116, 196), color[WHITE], use_font=FONT_LARGE)


def paint_score():
    print_at("%d  %s" % (highscore[0][0], highscore[0][1]), (0, 13),
             txt_color=color[YELLOW], center_flags=CENTER_X, anti_aliasing=0)
    for index in range(max_player):      # paint score for both player
        dim = print_at(str(scores[index]), (0, 0), color[WHITE], center_flags=NO_BLIT, use_font=FONT_SCORE)
        coords = (232 - dim[0] / X_SCALE, 16 + index * 11)
        print_at(str(scores[index]), coords, color[WHITE], use_font=FONT_SCORE)
        xco1 = min((17.0 + 3 * sparx_respawn), GAME_WIDTH/2)
        xco2 = max((238.0 - 3 * sparx_respawn), GAME_WIDTH/2)
        if xco1 < xco2:
            hal_draw_rect((xco1, 37), (xco2, 38), color[MIDRED])


def paint_sparx():
    for sparc in all_sparx:
        sprites = sprt_sparx
        if sparc[3]:
            sprites = sprt_supersparx
        show_sprite(sprites, sparc[:2])


def paint_playerpath():
    idx = DARKRED  # paint playerpath
    if move_mode[1] == MM_SPEED_FAST:
        idx = CYAN
    fuse_segment = find_intersect_index(players_path, fuse[:2], close=False)
    if len(fuse_segment) > 0:
        fuse_segment = fuse_segment[0] + 1
    else:
        fuse_segment = 0
    draw_list(players_path[:fuse_segment], color[DARKGREY], False)
    draw_list(players_path[fuse_segment:], color[idx], False)
    if fuse_segment < len(players_path):
        hal_draw_line(players_path[fuse_segment - 1], fuse[:2], color[DARKGREY])
        hal_draw_line(fuse[:2], players_path[fuse_segment], color[idx])
    if fuse[3]:  # paint fuse
        show_sprite(sprt_fuse, fuse[:2])


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


def paint_qix():
    for q in range(max_qix[current_player]):  # paint qixes
        for qix in qix_coords[current_player][q]:
            qix_lst = [(qix[0], qix[1]), (qix[2], qix[3])]
            draw_list(qix_lst, qix[4], False)


def paint_player():
    if is_dead:
        death_anim()
    else:
        pos = player_coords[current_player]
        player = [vector_add(pos, (-player_size, 0)), vector_add(pos, (0, player_size)),
                  vector_add(pos, (player_size, 0)), vector_add(pos, (0, -player_size))]
        draw_list(player, color[RED], True)
        if X_SCALE > 1.0:
            hal_draw_rect(vector_add(pos, (-1, -1)), vector_add(pos, (1, 1)), color[WHITE])
        else:
            hal_draw_rect(pos, vector_add(pos, (1, 1)), color[WHITE])  # add 1 pixel (pygame draws a 3x3 box on w=0)


def paint_playfield():
    for idx, iter_poly in enumerate(old_polys[current_player]):
        hal_fill_poly(iter_poly, old_poly_colors[current_player][idx])
    for iter_poly in old_polys[current_player]:
        draw_list(iter_poly, color[WHITE])
    draw_list(playfield[current_player], color[WHITE])


def paint_game():
    hal_blt(logo, (24, 16))
    paint_score()
    if game_mode == GM_GAME:
        paint_playfield()
        paint_claimed_and_lives()
        paint_playerpath()
        paint_player()
        paint_sparx()
        paint_qix()
    if game_mode == GM_GAMEOVER:
        if paint_game.wait_counter == -1:
            paint_game.wait_counter = frame_counter
        paint_playfield()
        print_at("CREDITS", (0, 22), txt_color=color[YELLOW], center_flags=CENTER_X, anti_aliasing=0)
        print_at("%02i" % credit, (0, 29), txt_color=color[YELLOW], center_flags=CENTER_X, anti_aliasing=0)
        hal_blt(game_over, game_over_coord)
        if (frame_counter - paint_game.wait_counter) > 2.5 * TPS:
            reset_playfield(current_player)
            set_game_mode(GM_HIGHSCORE_ENTRY)
    if game_mode == GM_HIGHSCORE_ENTRY:
        paint_claimed_and_lives()
        if scores[current_player] > highscore[-1][0]:
            paint_highscore_entry()
        else:
            set_game_mode(GM_HIGHSCORE)
    if game_mode == GM_HIGHSCORE:
        paint_game.wait_counter = getattr(paint_game, 'wait_counter', -1)  # waits for the wipe
        if paint_game.wait_counter == -1:
            paint_game.wait_counter = frame_counter
        paint_playfield()
        print_at("CREDITS", (0, 22), txt_color=color[YELLOW], center_flags=CENTER_X, anti_aliasing=0)
        print_at("%02i" % credit, (0, 29), txt_color=color[YELLOW], center_flags=CENTER_X, anti_aliasing=0)
        paint_highscore()
        if (frame_counter - paint_game.wait_counter) > 2.5 * TPS:
            set_game_mode(GM_GAMEOVER)


def show_sprite(img_stack, position):
    index = int(random.random() * len(img_stack))
    pos = (position[0] - img_stack[index].get_width() / 2 / X_SCALE,
           position[1] - img_stack[index].get_height() / 2 / Y_SCALE)
    hal_blt(img_stack[index], pos)


def revive_player():
    global dead_count_dir, player_coords, players_path, player_lives, current_player
    player_lives[current_player] -= 1
    if player_lives[current_player] > 0:
        dead_count_dir = -1
        reset_player_pos()
        reset_sparx(current_player)
    else:
        set_game_mode(GM_GAMEOVER)


def reset_player_pos():
    global players_path
    if len(players_path) > 0:
        player_coords[current_player] = players_path[0]
        players_path = list()
        move_mode[0] = MM_GRID
        fuse[3] = False


def generate_lines(center, width):
    retval = []
    for x in range(0, width):
        retval.append((vector_add(center, (x, -width)), vector_add(center, (-x, width))))
    for y in range(-width, width, 1):
        retval.append((vector_add(center, (width, y)), vector_add(center, (-width, -y))))
    for x in range(width, 0, -1):
        retval.append((vector_add(center, (x, width)), vector_add(center, (-x, -width))))
    random.shuffle(retval)
    return retval


def kill_player(qix_kill):
    global is_dead, dead_count_dir, killed_by_qix, dead_bubbles
    is_dead = True
    dead_count_dir = 1
    killed_by_qix = qix_kill
    if killed_by_qix:
        dead_bubbles = []
        index = 0
        if point_of_death == player_coords[current_player]:
            lines = generate_lines(player_coords[current_player], bubble_radius[1])
            dead_bubbles.append((player_coords[current_player], bubble_radius[1], 1, lines))
            dead_bubbles.append((player_coords[current_player], bubble_radius[0], 1, lines))
            index += 1
        for bubble in range(index, 3):
            center = vector_add(get_random_vector(dead_range), point_of_death)
            center = [int(center[0]), int(center[1])]
            lines = generate_lines(center, bubble_radius[1])
            dead_bubbles.append((center, bubble_radius[1], 1, lines))
            dead_bubbles.append((center, bubble_radius[0], 1, lines))


def move_fuse():
    if move_mode[0] != MM_GRID:  # handle fuse movement
        if fuse[2] > fuse_sleep:  # TODO: add up time player is waiting on path
            fuse[3] = True
            fuse[0:2] = calc_vertex_from_1d_path(players_path, fuse[0:2], 1, close=False)
            if vector_equal(fuse[0:2], player_coords[current_player]):  # fuse catches player!
                kill_player(False)
        else:
            fuse[2] += SKIP_TICKS


def move_qix():
    global frame_counter, qix_coords, point_of_death
    if frame_counter % 3 == 0:  # handle qix movement
        for q in range(max_qix[current_player]):
            qix = qix_move(q)
            qix.append(qix_change_color(q))
            qix_coords[current_player][q] = qix_coords[current_player][q][1:]
            qix_coords[current_player][q].append(qix)
            if move_mode[0] != MM_GRID:
                for line in qix_coords[current_player][q]:
                    collision = check_line_vs_poly(line[:2], line[2:], players_path, close=False, sort=False)
                    if len(collision) != 0:
                        point_of_death = get_first_collision(collision)
                        kill_player(True)


def move_sparx():
    for sparc in all_sparx:  # handle sparx movement
        if sparc[4]:  # supersparc has found players path before so it follows it
            sparc[0:2] = calc_vertex_from_1d_path(players_path, sparc[0:2], abs(sparc[2]), close=False)
            if sparc[0:2] == player_coords[current_player]:  # supersparx catches player!
                kill_player(False)
                sparc[0:2] = players_path[0]  # for debug mode reset to path start
        else:
            old_val = sparc[0:2]
            if len(sparc[5]) == 0:  # supersparc speed was flipped and is now in between old polys to find a way out
                search_polys = [playfield[current_player]]
                search_polys.extend(old_polys[current_player][::-1])
            else:
                search_polys = [sparc[6]]
            for single_poly in search_polys:
                sparc[0:2] = calc_vertex_from_1d_path(single_poly, sparc[0:2], sparc[2], close=True)
                if old_val != sparc[0:2]:
                    delta_path = cut_path(single_poly, old_val, sparc[0:2], sparc[2])
                    collision_with_player = find_intersect_index(delta_path, player_coords[current_player], close=False)
                    if len(collision_with_player) != 0:
                        kill_player(False)
                    resume_supersparx_normal(delta_path, sparc)
                    check_playerpath_supersparx(old_val, sparc)
                    break


def move_player(movement):
    global move_mode, players_path, half_frame_rate, old_polys, old_poly_colors, scores, playfield, fuse, \
        split_poly
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
                fuse = [players_path[0][0], players_path[0][1], 0, False]  # init fuse to hunt player
                possible_move = []  # allows to jump into next outer if for free roaming
            else:  # just overshooting..clip it to last corner point
                # intersecting movement with playfield
                tmp = [playfield[current_player][possible_move[0] - 1]]
                for mv in possible_move:
                    tmp.append(playfield[current_player][mv])
                tmp.append(playfield[current_player][(possible_move[-1] + 1) % len(playfield[current_player])])
                collision = check_line_vs_poly(player_coords[current_player], candidate, tmp, close=False)
                candidate = get_first_collision(collision, ignore_pt=player_coords[current_player])
        if move_mode[0] == MM_GRID:  # Check if a sparc is located between old_pos and current_pos
            delta_path_ccw = cut_path(playfield[current_player], player_coords[current_player], candidate, 1)
            delta_path_cw = cut_path(playfield[current_player], player_coords[current_player], candidate, -1)
            delta_path = delta_path_ccw
            if calc_1d_path(delta_path_ccw, False) > calc_1d_path(delta_path_cw, False):
                delta_path = delta_path_cw
            for sparc in all_sparx:
                collision_with_player = find_intersect_index(delta_path, sparc[0:2], close=False)
                if len(collision_with_player) != 0:
                    kill_player(False)
                    break

    if len(possible_move) == 0:  # Player is roaming a new line
        if fire_slow or fire_fast:
            fuse[3] = False
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
                fuse[:2] = players_path[0]
                # 1st: split playfield
                poly1, poly2 = split_polygon(playfield[current_player], list(players_path))
                old_playfield_area = abs(calc_area(playfield[current_player]))
                # 2nd: check which poly is new playfield(the one with the qix inside)
                if is_inside(poly1, qix_coords[current_player][0][0]):
                    playfield[current_player] = poly1
                    split_poly = poly2
                else:
                    playfield[current_player] = poly2
                    split_poly = poly1
                old_polys[current_player].append(split_poly)
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
                check_super_sparx_after_polysplit(players_path, all_sparx)
                players_path = []
                move_mode = [MM_GRID, None]
        else:  # in roaming mode only move if fast or slow button is pressed
            candidate = list(player_coords[current_player])
        # we are not allowing moving outside the playfield
    if not is_inside(playfield[current_player], candidate, strict=False):
        candidate = player_coords[current_player]
    return candidate


def check_playerpath_supersparx(old_val, sparc):
    if sparc[3] == 1 and len(players_path) > 0:
        collision_with_path = find_intersect_index([old_val, sparc[0:2]], players_path[0], close=False)
        if len(collision_with_path) != 0:  # found players path: set up sparx to follow it
            sparc[0:2] = players_path[0]
            sparc[4] = True  # this is a supersparc on players path


def check_super_sparx_after_polysplit(arg_players_path, arg_sparx):
    for sparc in arg_sparx:  # DONE: supersparx handling on old players Path: avoid sudden flip
        if sparc[4]:
            sparc[4] = False  # deactivate sparx on player's-path-mode
            poly_orientation = math.copysign(1, calc_area(arg_players_path))
            sparx_move = math.copysign(1, sparc[2])
            if poly_orientation != sparx_move:  # Case A or C? => Move sparx along last created poly to turn around
                sparc[5] = arg_players_path[0]  # resetting on start point
                sparc[6] = split_poly   # in Case C split_poly is reversed, so speed runs correctly
            else:  # Case B or D? = resume normal mode
                sparc[5] = []
                sparc[6] = []


def resume_supersparx_normal(delta_path, sparc):
    if len(sparc[5]) != 0:  # super sparx movement: check for ending flip mode
        flip_supersparx = find_intersect_index(delta_path, sparc[5], close=False)
        if len(flip_supersparx) != 0:
            sparc[4] = False
            sparc[5] = []
            sparc[6] = []


def qix_set_target():
    retval = []
    min_max = new_playfield
    while True:
        pt1 = [random.random()*(min_max[2][0] - min_max[0][0]),
               random.random()*(min_max[2][1] - min_max[0][1])]
        if is_inside(playfield[current_player], pt1, (0, 0)):
            break
    retval.append(pt1)
    # find point2 in max dist of around 1/4 of screen
    max_dist = (GAME_WIDTH / 4, GAME_HEIGHT / 4)
    loop_counter = 0
    while True:
        pt2 = get_random_vector(1.0, 1.0)
        pt2 = vector_add(pt1, (pt2[0] * random.random() * max_dist[0], pt2[1] * random.random() * max_dist[1]))
        if is_inside(playfield[current_player], pt2, (0, 0)):
            break
        loop_counter += 1
        if loop_counter > 5:
            max_dist = (max_dist[0] - 2, max_dist[1] - 2)
    retval.append(pt2)
    return retval


def calc_velocities(pt_start, pt_goal):
    dx, dy = vector_sub(pt_goal, pt_start)
    length = math.sqrt(dx ** 2 + dy ** 2)
    speed_factor = (random.random() * (qix_max_speed - qix_min_speed) + qix_min_speed)
    retval = [speed_factor * dx / length, speed_factor * dy / length]
    return retval


def check_collisions(points_to_check):
    pt1 = is_inside(playfield[current_player], points_to_check[0], (0, 0))
    pt2 = is_inside(playfield[current_player], points_to_check[1], (0, 0))
    return pt1, pt2


def qix_move(q):
    """
    Rules for qix movement:
    1. change direction every n frames
    2. check for collision of moving pts
    3. check for distance between dots.
    4. check for line collision.
    if any of the above occurred do:
      - move dot back to where it was before
      - create new direction and check again
    Calculates the new qix point by moving it by speed and checking for collision with playfield and adjusting
    speed and position accordingly
    :return: the final coordinates of new qix point
    """
    global qix_speed, qix_target, qix_change_counter
    retval = []
    dx = abs(qix_coords[current_player][q][-1][0] - qix_coords[current_player][q][-1][2])
    dy = abs(qix_coords[current_player][q][-1][1] - qix_coords[current_player][q][-1][3])
    qix_change_counter -= 1
    if dx > GAME_WIDTH/4 or dy > GAME_HEIGHT/4:
        pass
    loop_detect = 0
    vel_candidates = [(0, 0), (0, 0)]
    pt_candidates = [(0, 0), (0, 0)]
    if qix_change_counter < 0 or (dx**2 + dy**2) > ((GAME_WIDTH/4)**2 + (GAME_HEIGHT/4)**2):
        qix_target = qix_set_target()
        qix_change_counter = random.random() * (qix_max_change - qix_min_change) + qix_min_change
        qix_speed[q][0] = calc_velocities(qix_coords[current_player][q][-1][:2], qix_target[0])
        qix_speed[q][1] = calc_velocities(qix_coords[current_player][q][-1][2:], qix_target[1])
    vel_candidates[0] = (qix_speed[q][0][0], qix_speed[q][0][1])
    vel_candidates[1] = (qix_speed[q][1][0], qix_speed[q][1][1])
    pt_candidates[0] = vector_add(qix_coords[current_player][q][-1], vel_candidates[0])
    pt_candidates[1] = vector_add(qix_coords[current_player][q][-1][2:], vel_candidates[1])
    candidates = check_line_vs_poly(pt_candidates[0], pt_candidates[1], playfield[current_player], True)
    pt1_collision, pt2_collision = check_collisions(pt_candidates)
    while not pt1_collision \
            or not pt2_collision \
            or len(candidates) > 0:  # collision of moved point occurred?
        loop_detect += 1
        if loop_detect > 10:
            break
        qix_target = qix_set_target()
        qix_change_counter = random.random() * qix_max_change + qix_max_change
        vel_candidates[0] = calc_velocities(pt_candidates[0], qix_target[0])
        vel_candidates[1] = calc_velocities(pt_candidates[1], qix_target[1])

        pt_candidates[0] = vector_add(qix_coords[current_player][q][-1], vel_candidates[0])
        pt_candidates[1] = vector_add(qix_coords[current_player][q][-1][2:], vel_candidates[1])
        candidates = check_line_vs_poly(pt_candidates[0], pt_candidates[1], playfield[current_player], True)
        pt1_collision, pt2_collision = check_collisions(pt_candidates)
    qix_speed[q][0] = vel_candidates[0]
    qix_speed[q][1] = vel_candidates[1]
    retval.append(pt_candidates[0][0])
    retval.append(pt_candidates[0][1])
    retval.append(pt_candidates[1][0])
    retval.append(pt_candidates[1][1])
    return retval


def handle_movement():
    global move_mode, is_dead, dead_counter, dead_count_dir,\
        entry_accumulator, new_highscore_entry, entry_index, highscore
    if game_mode == GM_HIGHSCORE_ENTRY:
        entry_accumulator += 1
        if entry_accumulator > TPS / 8 or trigger_down or trigger_up or trigger_fast:  # 1/4 second for letter updates
            entry_accumulator = direction = 0
            if up:
                direction = 1
            if down:
                direction = -1
            if trigger_fast:
                entry_index += 1
                if entry_index < 3:
                    lst_entry = list(new_highscore_entry)
                    lst_entry[entry_index] = lst_entry[entry_index - 1]
                    new_highscore_entry = ''.join(lst_entry)
                else:
                    entry_index = 0
                    for index in range(len(highscore)):
                        if highscore[index][0] < scores[current_player]:
                            highscore.insert(index, [scores[current_player], new_highscore_entry])
                            highscore = highscore[:10]
                            new_highscore_entry = "..."
                            scores[current_player] = 0
                            with open(highscore_file, "w") as fp:
                                json.dump(highscore, fp)
                            break
                    set_game_mode(GM_HIGHSCORE)
            current_char = new_highscore_entry[entry_index]
            current_index = entry_chars.index(current_char)
            current_index = (current_index + direction) % len(entry_chars)
            lst_entry = list(new_highscore_entry)
            lst_entry[entry_index] = entry_chars[current_index]
            new_highscore_entry = ''.join(lst_entry)
    if game_mode == GM_GAME:
        if is_dead:
            dead_counter += dead_count_dir * SKIP_TICKS
            if dead_counter <= 0:
                dead_count_dir = 0
                is_dead = False
            if dead_count_dir < 0:
                move_qix()
        else:
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
                candidate = move_player(movement)
                if move_mode[0] == MM_VERTICAL or move_mode[0] == MM_HORIZONTAL:
                    if candidate == player_coords[current_player] and not half_frame_rate:
                        move_fuse()
                player_coords[current_player] = candidate
                move_sparx()
                move_qix()


def reset_playfield(index_player):
    global playfield, old_polys, old_poly_colors, players_path
    playfield[index_player] = new_playfield
    old_polys[index_player] = []  # the polys which were the borders before (need for sparx movement)
    old_poly_colors[index_player] = []  # the colors of the polys (fast or slow)
    players_path = []  # the path the player will draw on screen


def reset():
    global current_player, credit, player_lives, player_coords, move_mode, fuse, max_qix, qix_coords, \
        trigger_up, trigger_down, trigger_fast, fire_slow, fire_fast, up, down, left, right, \
        is_dead, dead_counter, dead_count_dir, scores
    current_player = 0
    credit -= 1
    reset_playfield(0)
    player_lives = [start_player_lives, start_player_lives]
    player_coords = [player_start, player_start]
    move_mode = [MM_GRID, MM_SPEED_SLOW]
    fuse = [0, 0, 0, False]  # fuse hunts player, if he draws a line and stops[x,y,sleep_timer,visible]
    scores = [0, 0]
    reset_sparx(0)
    max_qix = [1, 1]
    qix_coords = [[[], []], [[], []]]  # 2 qixes with x x/y coordinate of qix
    init_qix(0)
    set_game_mode(GM_GAME)
    trigger_up = trigger_down = trigger_fast = fire_slow = fire_fast = up = down = left = right = is_dead = False
    dead_counter = calc_max_exploding_line_steps()
    dead_count_dir = -1
    is_dead = True


def init_qix(index_player):
    global qix_coords, qix_speed
    for q in range(max_qix[index_player]):
        qix_speed[q] = [get_random_vector(15, 5), get_random_vector(15, 5)]
        qix_coords[index_player][q] = []
        for index in range(0, 7):
            qix_coords[index_player][q].append(
                [(GAME_WIDTH * 0.45) + index * qix_speed[q][0][0], (GAME_HEIGHT * 0.45) + index * qix_speed[q][0][1],
                 (GAME_WIDTH * 0.45) + index * qix_speed[q][1][0], (GAME_HEIGHT * 0.45) + index * qix_speed[q][1][1],
                 color[qix_color_index[q]]])
            qix_change_color(q)
        qix_speed[q] = [get_random_vector(15, 5), get_random_vector(15, 5)]


def init():
    global window_surface, screen, logo, fonts, game_over, active_live, inactive_live, player_lives, player_coords,\
        move_mode, fuse, sprt_fuse, sprt_sparx, sprt_supersparx, highscore, max_qix, qix_coords
    pygame.init()
    window_surface = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
    screen = pygame.Surface((WIDTH, HEIGHT))
    fonts = [pygame.font.Font("data/qix-small.ttf", int(8.0 * Y_SCALE)),
             pygame.font.Font("data/qix-large.ttf", int(8.0 * Y_SCALE)),
             pygame.font.Font("data/qix-large.ttf", int(10.0 * Y_SCALE))]
    logo, _ = hal_load_image(os.path.join('data', 'qix_logo.png'))
    logo = pygame.transform.scale(logo, (int(56.0 * X_SCALE), int(20 * Y_SCALE)))
    game_over, _ = hal_load_image(os.path.join('data', 'qix_game_over.png'))
    game_over = pygame.transform.scale(game_over,  (int(91 * X_SCALE), int(17 * Y_SCALE)))
    active_live, _ = hal_load_image(os.path.join('data', 'qix_live_w.png'))
    active_live = pygame.transform.scale(active_live, (int(3.0 * X_SCALE), int(3.0 * Y_SCALE)))
    inactive_live, _ = hal_load_image(os.path.join('data', 'qix_live_r.png'))
    inactive_live = pygame.transform.scale(inactive_live,  (int(3.0 * X_SCALE), int(3.0 * Y_SCALE)))
    for index in range(8):
        tmp, size = hal_load_image(os.path.join('data', 'fuse', 'fuse_%d.png' % (index + 1)))
        tmp = pygame.transform.scale(tmp, (max(int(size[2] * X_SCALE), 1), max(int(size[3] * Y_SCALE), 1)))
        tmp.set_colorkey(Color(0))
        sprt_fuse.append(tmp)
    for index in range(14):
        tmp, size = hal_load_image(os.path.join('data', 'sparx', 'sparx_%02d.png' % (index + 1)))
        tmp = pygame.transform.scale(tmp, (max(int(size[2] * X_SCALE), 1), max(int(size[3] * Y_SCALE), 1)))
        tmp.set_colorkey(Color(0))
        sprt_sparx.append(tmp)
    for index in range(16):
        tmp, size = hal_load_image(os.path.join('data', 'super_sparx', 'super_sparx_%02d.png' % (index + 1)))
        tmp = pygame.transform.scale(tmp, (max(int(size[2] * X_SCALE), 1), max(int(size[3] * Y_SCALE), 1)))
        tmp.set_colorkey(Color(0))
        sprt_supersparx.append(tmp)
    if os.path.exists(highscore_file):
        try:
            with open(highscore_file) as fp:
                highscore = json.load(fp)
        except OSError:
            highscore = [(30000, "QIX") for i in range(10)]
    set_game_mode(GM_GAMEOVER)
    reset_playfield(0)
    player_lives = [start_player_lives, start_player_lives]
    player_coords = [player_start, player_start]
    move_mode = [MM_GRID, MM_SPEED_SLOW]
    fuse = [0, 0, 0, False]  # fuse hunts player, if he draws a line and stops[x,y,sleep_timer,visible]
    reset_sparx(0)
    max_qix = [1, 1]
    qix_coords = [[[], []], [[], []]]  # 2 qixes with x x/y coordinate of qix
    init_qix(0)
    init_qix(1)


def press_key(key):
    global pressed_keys, left, right, up, down, fire_slow, fire_fast, trigger_up, trigger_down, trigger_fast, credit
    if key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
        pressed_keys.append(key)
    if key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
        left = right = up = down = False
    if key == pygame.K_LEFT and pygame.K_RIGHT not in pressed_keys:
        left = True
    if key == pygame.K_RIGHT and pygame.K_LEFT not in pressed_keys:
        right = True
    if key == pygame.K_UP and pygame.K_DOWN not in pressed_keys:
        up = trigger_up = True
    if key == pygame.K_DOWN and pygame.K_UP not in pressed_keys:
        down = trigger_down = True
    if key == pygame.K_LALT:
        fire_slow = True
    if key == pygame.K_LCTRL:
        fire_fast = trigger_fast = True
    if key == pygame.K_5:
        credit += 1
    if key == pygame.K_6:
        credit += 2
    if credit > 30:
        credit = 30


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
    if key == pygame.K_1:  # "1"-key
        if credit > 0 and game_mode in [GM_GAMEOVER, GM_HIGHSCORE]:
            reset()
    if key == pygame.K_2:  # "2"-key
        if credit > 1 and game_mode in [GM_GAMEOVER, GM_HIGHSCORE]:
            reset()


def gameloop():  # https://dewitters.com/dewitters-gameloop/
    global frame_counter, trigger_up, trigger_down, trigger_fast
    next_game_tick = get_real_time() - 1
    one_sec_tick = get_time() + 1000
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
            trigger_up = trigger_down = trigger_fast = False
            frame_counter += 1
            if get_time() - one_sec_tick > 0:
                if not is_dead and player_lives[current_player] > 0 and game_mode == GM_GAME:
                    tick_sparc_respawn()
                one_sec_tick = get_time() + 1000
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
    print("<5>, <6> throw in 1 / 2 credits")
    print("<1>, <2> start one player game")
    print("<CTRL> is FAST button")
    print("<ALT> is SLOW button")
    print("Cursor keys are Joystick")
    init()
    gameloop()
