"""Render chess positions in Blender with exact labels: piece boxes, board
corners and the position. Runs inside Blender (5.x), not in the venv:

    blender -b -P render_boards.py -- --count 2000 --out datasets/render --seed 1
    blender -b -P render_boards.py -- --count 12 --out runs/render_preview --preview

Why: on a board the models have not seen (Fenit board B) the piece model
reads the piece type wrong on one piece in twelve, most of all pawns seen
from straight above and glossy dark pieces, and takes brown "white" pieces
for black. The real training photos are a dozen piece sets, nearly all seen
from a player's seat. Here every image has its own piece set (turned
profiles with random proportions, a cut-out knight), its own board, floor,
light and camera, from a low side view to straight above.

Every image gets:
  pieces/labels/<stem>.txt   one box per piece, class ids of chess_pipeline.PIECE_CLASSES
  corners/labels/<stem>.txt  the playing area's corners (a1, a8, h8, h1), when
                             all four are inside the picture
  positions.jsonl            stem, FEN placement, camera, for scoring and debugging
in the `native` layout build_dataset.py merges (source `render` in
sources.yaml). Boxes cover the whole piece, also where another piece hides
part of it, as in the hand-labelled datasets.

Positions come from --positions (a positions.json written by build_dataset.py
merge; its train split is used), thinned at random so that endgames are as
common as openings, or are made up when that file is not given.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

PIECE_CLASSES = [
    "white_king", "white_queen", "white_rook", "white_bishop", "white_knight", "white_pawn",
    "black_king", "black_queen", "black_rook", "black_bishop", "black_knight", "black_pawn",
]
KIND = {"k": "king", "q": "queen", "r": "rook", "b": "bishop", "n": "knight", "p": "pawn"}
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


# --------------------------------------------------------------------------
# Piece sets. Lengths are in squares; a piece stands on z = 0, axis along z.

def lathe(name, profile, segments=28):
    """A solid of revolution from (radius, height) points, bottom to top. The
    first and last points should be on the axis (radius 0)."""
    verts, faces = [], []
    rings = []
    for r, z in profile:
        if r <= 1e-6:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            start = len(verts)
            for i in range(segments):
                a = 2 * math.pi * i / segments
                verts.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(list(range(start, start + segments)))
    for lower, upper in zip(rings, rings[1:]):
        if len(lower) == 1 and len(upper) == 1:
            continue
        for i in range(segments):
            j = (i + 1) % segments
            if len(lower) == 1:
                faces.append((lower[0], upper[j], upper[i]))
            elif len(upper) == 1:
                faces.append((lower[i], lower[j], upper[0]))
            else:
                faces.append((lower[i], lower[j], upper[j], upper[i]))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def box_mesh(name, size, centre):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * size[0] + centre[0], v.co.y * size[1] + centre[1],
                       v.co.z * size[2] + centre[2]))
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def sphere_mesh(name, radius, centre):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=radius)
    for v in bm.verts:
        v.co = v.co + Vector(centre)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def knight_head(name, width, height, thickness, z0, rng):
    """A horse's head cut out of a slab: a side outline (facing +x) given
    thickness along y and rounded off."""
    outline = [(-0.40, 0.00), (0.36, 0.00), (0.30, 0.20), (0.14, 0.38), (0.36, 0.44),
               (0.54, 0.52), (0.56, 0.66), (0.40, 0.80), (0.22, 0.93), (0.12, 1.00),
               (0.02, 0.90), (-0.20, 0.82), (-0.38, 0.56), (-0.44, 0.24)]
    jitter = 0.03
    pts = [((x + rng.uniform(-jitter, jitter)) * width,
            z0 + max(y + rng.uniform(-jitter, jitter), 0.0) * height) for x, y in outline]
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    front = [bm.verts.new((x, -thickness / 2, z)) for x, z in pts]
    back = [bm.verts.new((x, thickness / 2, z)) for x, z in pts]
    bm.faces.new(front)
    bm.faces.new(list(reversed(back)))
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((front[j], front[i], back[i], back[j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=thickness * rng.uniform(0.15, 0.3),
                    segments=2, affect="EDGES")
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def join(name, meshes):
    """One mesh out of several."""
    bm = bmesh.new()
    for m in meshes:
        bm.from_mesh(m)
        bpy.data.meshes.remove(m)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = True
    return mesh


def body(base_r, top_z, st):
    """Profile of the foot and stem that all turned pieces share, from the
    middle of the underside up to the neck at top_z."""
    foot = st["foot"] * top_z
    waist = st["waist"] * base_r
    pts = [(0.0, 0.0), (base_r, 0.0), (base_r, foot * 0.55), (base_r * 0.93, foot)]
    if st["step"]:
        pts += [(base_r * 0.78, foot * 1.05), (base_r * 0.78, foot * 1.45), (base_r * 0.70, foot * 1.55)]
    belly = foot + (top_z - foot) * st["belly_at"]
    pts += [(base_r * st["belly"], belly), (waist * 1.15, belly + (top_z - belly) * 0.45),
            (waist, top_z)]
    return pts


def collar(r, z, thick):
    return [(r * 0.75, z), (r, z + thick * 0.4), (r, z + thick * 0.6), (r * 0.7, z + thick)]


def make_set(rng):
    """Six meshes, one per kind, in one random style."""
    st = dict(
        foot=rng.uniform(0.08, 0.16), step=rng.random() < 0.5,
        belly=rng.uniform(0.55, 0.85), belly_at=rng.uniform(0.12, 0.3),
        waist=rng.uniform(0.28, 0.45), collar=rng.uniform(0.55, 0.8),
    )
    size = rng.uniform(0.82, 1.12)     # how tall the set is for its squares
    girth = rng.uniform(0.85, 1.12)
    tall = {"pawn": rng.uniform(0.82, 1.0), "rook": rng.uniform(0.98, 1.15),
            "knight": rng.uniform(1.1, 1.3), "bishop": rng.uniform(1.3, 1.5),
            "queen": rng.uniform(1.5, 1.75), "king": rng.uniform(1.7, 1.98)}
    wide = {"pawn": 0.26, "rook": 0.30, "knight": 0.31, "bishop": 0.31, "queen": 0.345, "king": 0.36}
    out = {}
    for kind in KIND.values():
        h, r = tall[kind] * size, wide[kind] * girth
        w = st["waist"] * r
        if kind == "pawn":
            ball = r * rng.uniform(0.52, 0.68)
            neck = h - 2 * ball * rng.uniform(0.85, 0.95)
            c = r * st["collar"]
            pts = body(r, neck - 0.04 * h, st) + collar(c, neck - 0.04 * h, 0.05 * h)
            cz = h - ball
            pts += [(ball * math.sin(a), cz - ball * math.cos(a))
                    for a in np.linspace(0.6, math.pi, 8)]
            mesh = lathe(kind, pts)
        elif kind == "rook":
            top_r = r * rng.uniform(0.68, 0.85)
            wall = h * rng.uniform(0.16, 0.24)
            pts = body(r, h - wall * 1.6, st)
            pts[-1] = (max(w * 1.6, top_r * 0.75), pts[-1][1])
            pts += [(top_r, h - wall * 1.3), (top_r, h), (top_r * 0.72, h),
                    (top_r * 0.72, h - wall * 0.45), (0.0, h - wall * 0.45)]
            parts = [lathe(kind, pts)]
            if rng.random() < 0.8:  # battlements: blocks on the rim
                n = rng.choice([4, 5, 6])
                for i in range(n):
                    a = 2 * math.pi * i / n
                    rr = top_r * 0.86
                    parts.append(box_mesh("m", (top_r * 0.5, top_r * 0.3, wall * 0.35),
                                          (rr * math.cos(a), rr * math.sin(a), h + wall * 0.15)))
                    v = parts[-1].vertices
                    for vert in v:  # turn the block to face outwards
                        x, y = vert.co.x - rr * math.cos(a), vert.co.y - rr * math.sin(a)
                        vert.co.x = rr * math.cos(a) + x * math.cos(a + math.pi / 2) - y * math.sin(a + math.pi / 2)
                        vert.co.y = rr * math.sin(a) + x * math.sin(a + math.pi / 2) + y * math.cos(a + math.pi / 2)
            mesh = join(kind, parts)
        elif kind == "bishop":
            neck = h * rng.uniform(0.56, 0.64)
            c = r * st["collar"]
            head_r = r * rng.uniform(0.48, 0.62)
            tip = h * 0.045
            pts = body(r, neck, st) + collar(c, neck, 0.04 * h)
            top = h - 2 * tip
            for t in np.linspace(0.0, 1.0, 9):
                z = neck + 0.04 * h + (top - neck - 0.04 * h) * t
                rad = head_r * math.sin(math.pi * (0.25 + 0.75 * t)) ** 0.8 if t < 1 else tip * 0.5
                pts.append((max(rad, tip * 0.5), z))
            pts += [(tip, top + tip), (tip * 0.7, top + 1.7 * tip), (0.0, h)]
            mesh = lathe(kind, pts)
        elif kind == "queen":
            neck = h * rng.uniform(0.58, 0.66)
            c = r * st["collar"]
            crown = r * rng.uniform(0.62, 0.8)
            pts = body(r, neck, st) + collar(c, neck, 0.035 * h)
            rim = h * rng.uniform(0.86, 0.9)
            pts += [(w * 1.2, neck + 0.05 * h), (crown * 0.75, rim - 0.06 * h), (crown, rim),
                    (crown * 0.8, rim + 0.01 * h), (crown * 0.5, rim + 0.035 * h),
                    (0.05 * h, h - 0.07 * h), (0.045 * h, h - 0.03 * h), (0.0, h)]
            parts = [lathe(kind, pts)]
            if rng.random() < 0.7:  # points of the crown
                n = rng.choice([6, 8, 10])
                for i in range(n):
                    a = 2 * math.pi * i / n
                    parts.append(sphere_mesh("p", crown * 0.14,
                                             (crown * 0.92 * math.cos(a), crown * 0.92 * math.sin(a), rim + crown * 0.06)))
            mesh = join(kind, parts)
        elif kind == "king":
            neck = h * rng.uniform(0.52, 0.6)
            c = r * st["collar"]
            crown = r * rng.uniform(0.6, 0.75)
            cross = h * rng.uniform(0.12, 0.17)
            rim = h - cross - 0.05 * h
            pts = body(r, neck, st) + collar(c, neck, 0.035 * h)
            pts += [(w * 1.2, neck + 0.05 * h), (crown * 0.8, rim - 0.05 * h), (crown, rim),
                    (crown * 0.85, rim + 0.02 * h), (crown * 0.3, rim + 0.05 * h), (0.0, rim + 0.05 * h)]
            arm = cross * rng.uniform(0.2, 0.3)
            parts = [lathe(kind, pts),
                     box_mesh("c1", (arm, arm, cross), (0, 0, h - cross / 2)),
                     box_mesh("c2", (cross * 0.75, arm, arm), (0, 0, h - cross * 0.38))]
            mesh = join(kind, parts)
        else:  # knight
            plinth = h * rng.uniform(0.2, 0.28)
            pts = body(r, plinth, st)
            pts[-1] = (r * 0.7, plinth)
            pts.append((0.0, plinth))
            head = knight_head("h", r * rng.uniform(1.5, 1.9), h - plinth * 0.9,
                               r * rng.uniform(0.6, 0.85), plinth * 0.9, rng)
            mesh = join(kind, [lathe(kind, pts), head])
        for poly in mesh.polygons:
            poly.use_smooth = True
        out[kind] = mesh
    return out


# --------------------------------------------------------------------------
# Materials, board, floor, light

def material(name, colour, roughness, rng, coat=0.0, texture=None, scale=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
    if texture is not None:
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = texture
        tex.interpolation = "Linear"
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def mottled(name, a, b, roughness, rng, kind):
    """A floor or table: two colours mixed by a procedural pattern."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = roughness
    coords = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0, 0, rng.uniform(0, math.pi))
    links.new(coords.outputs["Object"], mapping.inputs["Vector"])
    if kind == "tiles":
        tex = nodes.new("ShaderNodeTexBrick")
        tex.inputs["Scale"].default_value = rng.uniform(0.08, 0.5)
        tex.inputs["Mortar Size"].default_value = rng.uniform(0.002, 0.02)
        tex.offset = rng.choice([0.0, 0.5])
        tex.inputs["Color1"].default_value = (*a, 1)
        tex.inputs["Color2"].default_value = (*[min(c * rng.uniform(0.9, 1.1), 1) for c in a], 1)
        tex.inputs["Mortar"].default_value = (*b, 1)
        colour = tex.outputs["Color"]
    elif kind == "checker":
        tex = nodes.new("ShaderNodeTexChecker")
        tex.inputs["Scale"].default_value = rng.uniform(0.1, 0.6)
        tex.inputs["Color1"].default_value = (*a, 1)
        tex.inputs["Color2"].default_value = (*b, 1)
        colour = tex.outputs["Color"]
    else:
        tex = nodes.new("ShaderNodeTexWave" if kind == "wood" else "ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = rng.uniform(0.15, 1.5) if kind == "wood" else rng.uniform(0.3, 6)
        if kind == "wood":
            tex.inputs["Distortion"].default_value = rng.uniform(1, 6)
            tex.inputs["Detail"].default_value = 3
        ramp = nodes.new("ShaderNodeMixRGB") if hasattr(bpy.types, "ShaderNodeMixRGB") else None
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs["A"].default_value = (*a, 1)
        mix.inputs["B"].default_value = (*b, 1)
        if ramp is not None:
            nodes.remove(ramp)
        links.new(tex.outputs["Fac"] if "Fac" in tex.outputs else tex.outputs[0], mix.inputs["Factor"])
        colour = mix.outputs["Result"]
    links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    links.new(colour, bsdf.inputs["Base Color"])
    return mat


SQUARES = [  # light, dark
    ((0.93, 0.86, 0.70), (0.45, 0.27, 0.14)), ((0.92, 0.92, 0.84), (0.30, 0.45, 0.25)),
    ((0.90, 0.90, 0.90), (0.08, 0.08, 0.08)), ((0.85, 0.75, 0.55), (0.35, 0.18, 0.08)),
    ((0.95, 0.93, 0.85), (0.25, 0.20, 0.18)), ((0.80, 0.65, 0.42), (0.28, 0.14, 0.07)),
    ((0.88, 0.90, 0.93), (0.25, 0.38, 0.55)), ((0.93, 0.88, 0.75), (0.55, 0.20, 0.15)),
    ((0.75, 0.75, 0.72), (0.35, 0.35, 0.35)), ((0.96, 0.90, 0.78), (0.62, 0.42, 0.25)),
]
WHITE_SIDE = [(0.92, 0.88, 0.76), (0.95, 0.95, 0.93), (0.85, 0.70, 0.45), (0.70, 0.48, 0.26),
              (0.55, 0.32, 0.16), (0.80, 0.60, 0.40), (0.88, 0.80, 0.60), (0.62, 0.30, 0.18)]
BLACK_SIDE = [(0.015, 0.015, 0.015), (0.03, 0.03, 0.035), (0.10, 0.05, 0.03), (0.16, 0.08, 0.04),
              (0.08, 0.08, 0.09), (0.05, 0.03, 0.02), (0.12, 0.03, 0.03)]


def vary(colour, rng, amount=0.08):
    return tuple(min(max(c * rng.uniform(1 - amount, 1 + amount) + rng.uniform(-0.02, 0.02), 0.0), 1.0)
                 for c in colour)


def board_texture(rng, border, px=64):
    """The board from above as an image, a1 a dark square at the bottom left
    of the playing area, with a margin `border` squares wide. Returns the
    image and the two square colours."""
    light, dark = (vary(c, rng) for c in rng.choice(SQUARES))
    if rng.random() < 0.12:  # a few boards with unusual, weak or strong contrast
        dark = vary((rng.random() * 0.6, rng.random() * 0.6, rng.random() * 0.6), rng)
    b = int(round(border * px))
    side = 8 * px + 2 * b
    img = np.zeros((side, side, 4), np.float32)
    img[..., 3] = 1.0
    margin = vary(rng.choice([light, dark, (0.35, 0.2, 0.1), (0.1, 0.1, 0.1), (0.85, 0.85, 0.82),
                              (0.5, 0.33, 0.18)]), rng)
    img[..., :3] = margin
    for f in range(8):
        for r in range(8):
            colour = dark if (f + r) % 2 == 0 else light
            noise = rng.uniform(0.96, 1.04)
            img[b + r * px: b + (r + 1) * px, b + f * px: b + (f + 1) * px, :3] = [c * noise for c in colour]
    if b >= 4 and rng.random() < 0.5:  # a line around the playing area
        line = vary(rng.choice([light, dark, (0.05, 0.05, 0.05), (0.9, 0.9, 0.9)]), rng)
        gap, wd = rng.randint(1, max(1, b // 3)), rng.randint(1, max(1, b // 6))
        lo, hi = b - gap - wd, side - b + gap
        for a, c in ((lo, lo + wd), (hi, hi + wd)):
            img[a:c, lo:hi + wd, :3] = line
            img[lo:hi + wd, a:c, :3] = line
    image = bpy.data.images.new("board", side, side)
    image.pixels.foreach_set(img.ravel())   # row 0 is the bottom row: rank 1
    image.pack()
    return image, light, dark


def build_scene(rng, placement):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    col = scene.collection

    # Board: the playing area is [0, 8] x [0, 8], a1's corner at the origin,
    # files along +x, ranks along +y, top surface at z = 0.
    border = rng.choice([0.0, rng.uniform(0.05, 0.3), rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.9)])
    image, light, dark = board_texture(rng, border)
    thick = rng.choice([0.01, rng.uniform(0.05, 0.25), rng.uniform(0.25, 0.6)])
    lo, hi = -border, 8 + border
    mesh = bpy.data.meshes.new("board")
    verts = [(lo, lo, 0), (hi, lo, 0), (hi, hi, 0), (lo, hi, 0),
             (lo, lo, -thick), (hi, lo, -thick), (hi, hi, -thick), (lo, hi, -thick)]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    mesh.from_pydata(verts, [], faces)
    uv = mesh.uv_layers.new(name="uv")
    corner_uv = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
    for poly in mesh.polygons:
        for loop in poly.loop_indices:
            v = mesh.loops[loop].vertex_index
            uv.data[loop].uv = corner_uv.get(v, (0.01, 0.01))  # the sides take the margin colour
    board = bpy.data.objects.new("board", mesh)
    mesh.materials.append(material("board", (1, 1, 1), rng.choice([rng.uniform(0.05, 0.3), rng.uniform(0.3, 0.9)]),
                                   rng, coat=rng.choice([0.0, 0.0, rng.uniform(0.2, 1.0)]), texture=image))
    col.objects.link(board)

    # Floor or table under it.
    ground_mesh = bpy.data.meshes.new("ground")
    g = 80
    ground_mesh.from_pydata([(-g, -g, -thick), (g, -g, -thick), (g, g, -thick), (-g, g, -thick)], [], [(0, 1, 2, 3)])
    ground = bpy.data.objects.new("ground", ground_mesh)
    base = vary(rng.choice([(0.7, 0.68, 0.62), (0.55, 0.38, 0.22), (0.35, 0.22, 0.12), (0.8, 0.8, 0.8),
                            (0.25, 0.25, 0.27), (0.6, 0.62, 0.55), (0.75, 0.6, 0.4), (0.12, 0.12, 0.12),
                            (0.3, 0.4, 0.3), (0.5, 0.15, 0.12), (0.2, 0.25, 0.4)]), rng, 0.2)
    other = vary(base, rng, 0.35)
    ground_mesh.materials.append(mottled(
        "ground", base, other, rng.choice([rng.uniform(0.03, 0.25), rng.uniform(0.3, 0.95)]), rng,
        rng.choice(["tiles", "tiles", "wood", "wood", "noise", "noise", "checker"])))
    col.objects.link(ground)

    # Pieces.
    meshes = make_set(rng)
    white = vary(rng.choice(WHITE_SIDE), rng)
    black = vary(rng.choice(BLACK_SIDE), rng, 0.3)
    gloss = rng.choice([rng.uniform(0.03, 0.2), rng.uniform(0.2, 0.5), rng.uniform(0.5, 0.9)])
    coat = rng.choice([0.0, rng.uniform(0.3, 1.0)])
    mats = {True: material("white", white, min(gloss * rng.uniform(0.8, 1.3), 1), rng, coat),
            False: material("black", black, min(gloss * rng.uniform(0.8, 1.3), 1), rng, coat)}
    per_colour = {}
    for is_white in (True, False):
        for kind, m in meshes.items():
            copy = m.copy()
            copy.materials.append(mats[is_white])
            per_colour[(kind, is_white)] = copy
    sloppy = rng.choice([0.04, 0.1, 0.18])   # how far off-centre pieces stand
    pieces = []
    rows = placement.split("/")
    for row_index, row in enumerate(rows):
        rank, file = 7 - row_index, 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
                continue
            kind, is_white = KIND[ch.lower()], ch.isupper()
            obj = bpy.data.objects.new(f"{ch}_{file}{rank}", per_colour[(kind, is_white)])
            obj.location = (file + 0.5 + rng.uniform(-sloppy, sloppy), rank + 0.5 + rng.uniform(-sloppy, sloppy), 0)
            facing = (math.pi / 2 if is_white else -math.pi / 2)  # knights look at the other side
            obj.rotation_euler = (0, 0, facing + (rng.uniform(-0.6, 0.6) if rng.random() < 0.8
                                                  else rng.uniform(0, 2 * math.pi)))
            col.objects.link(obj)
            pieces.append((obj, f"{'white' if is_white else 'black'}_{kind}"))
            file += 1

    # Light: a sky colour, a sun or a ceiling lamp or two.
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    tint = rng.uniform(0.85, 1.0)
    bg.inputs["Color"].default_value = (tint, tint * rng.uniform(0.92, 1.0), tint * rng.uniform(0.8, 1.0), 1)
    bg.inputs["Strength"].default_value = rng.uniform(0.1, 0.7)
    scene.world = world
    if rng.random() < 0.6:
        sun = bpy.data.lights.new("sun", "SUN")
        sun.energy = rng.uniform(0.4, 3.0)
        sun.angle = rng.uniform(0.02, 0.6)
        ob = bpy.data.objects.new("sun", sun)
        ob.rotation_euler = (rng.uniform(0, 1.1), 0, rng.uniform(0, 2 * math.pi))
        col.objects.link(ob)
    for _ in range(rng.choice([0, 1, 1, 2, 3])):
        lamp = bpy.data.lights.new("lamp", "AREA")
        lamp.size = rng.uniform(1, 8)
        dist = rng.uniform(10, 30)
        lamp.energy = rng.uniform(100, 600) * (dist / 15) ** 2
        ob = bpy.data.objects.new("lamp", lamp)
        ob.location = (4 + rng.uniform(-14, 14), 4 + rng.uniform(-14, 14), dist)
        aim = Vector((4, 4, 0)) - Vector(ob.location)
        ob.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
        col.objects.link(ob)
    return scene, pieces, border


def place_camera(scene, rng, border, size):
    w, h = size
    cam_data = bpy.data.cameras.new("camera")
    cam_data.sensor_fit = "AUTO"
    cam_data.sensor_width = 36
    cam_data.lens = rng.uniform(22, 50)
    cam = bpy.data.objects.new("camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100

    # From a low side view to straight above, from any side of the board.
    r = rng.random()
    elevation = (rng.uniform(22, 40) if r < 0.2 else rng.uniform(40, 70) if r < 0.7
                 else rng.uniform(70, 90))
    azimuth = rng.uniform(0, 360)
    fill = rng.uniform(0.6, 1.15)      # how much of the short side the board takes up
    extent = (8 + 2 * border) * 1.15   # seen straight on; more from a corner
    fov = 2 * math.atan(36 * min(w, h) / max(w, h) / (2 * cam_data.lens))
    distance = extent / 2 / math.tan(fov / 2) / fill
    e, a = math.radians(elevation), math.radians(azimuth)
    target = Vector((4 + rng.uniform(-1.2, 1.2), 4 + rng.uniform(-1.2, 1.2), 0))
    cam.location = target + distance * Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.rotation_euler.rotate_axis("Z", math.radians(rng.uniform(-8, 8) if rng.random() < 0.8
                                                     else rng.uniform(-180, 180)))
    return cam, dict(elevation=round(elevation, 1), azimuth=round(azimuth, 1), lens=round(cam_data.lens, 1))


def project(scene, cam, point):
    """World point -> (x, y) as fractions of the picture, origin top left."""
    v = world_to_camera_view(scene, cam, Vector(point))
    return v.x, 1.0 - v.y, v.z


def labels(scene, cam, pieces):
    bpy.context.view_layer.update()
    corners = [project(scene, cam, p)[:2] for p in ((0, 0, 0), (0, 8, 0), (8, 8, 0), (8, 0, 0))]
    boxes = []
    for obj, cls in pieces:
        m = obj.matrix_world
        pts = [project(scene, cam, m @ v.co) for v in obj.data.vertices]
        if any(z <= 0 for _, _, z in pts):
            continue
        x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
        y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
        cx0, cx1, cy0, cy1 = max(x0, 0.0), min(x1, 1.0), max(y0, 0.0), min(y1, 1.0)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) < 0.4 * (x1 - x0) * (y1 - y0):
            continue  # mostly outside the picture
        boxes.append((PIECE_CLASSES.index(cls), (cx0 + cx1) / 2, (cy0 + cy1) / 2, cx1 - cx0, cy1 - cy0))
    return corners, boxes


# --------------------------------------------------------------------------
# Positions

def thin(placement, rng):
    """The position with some pieces taken off, so that sparse boards are as
    common as full ones. Kings stay."""
    keep = rng.choice([1.0, 1.0, rng.uniform(0.5, 0.9), rng.uniform(0.1, 0.5)])
    rows = []
    for row in placement.split("/"):
        out, empty = "", 0
        for ch in row:
            if ch.isdigit():
                empty += int(ch)
            elif ch.lower() != "k" and rng.random() > keep:
                empty += 1
            else:
                out += (str(empty) if empty else "") + ch
                empty = 0
        rows.append(out + (str(empty) if empty else ""))
    return "/".join(rows)


def made_up(rng):
    """A random legal-looking position: both kings, no pawns on the end ranks."""
    squares = [(f, r) for f in range(8) for r in range(8)]
    rng.shuffle(squares)
    board = {squares.pop(): "K", squares.pop(): "k"}
    for _ in range(rng.randint(0, 28)):
        ch = rng.choice("PPPPNBRQ")
        ch = ch if rng.random() < 0.5 else ch.lower()
        sq = squares.pop()
        if ch.lower() == "p" and sq[1] in (0, 7):
            continue
        board[sq] = ch
    rows = []
    for rank in range(7, -1, -1):
        out, empty = "", 0
        for file in range(8):
            ch = board.get((file, rank))
            if ch is None:
                empty += 1
            else:
                out += (str(empty) if empty else "") + ch
                empty = 0
        rows.append(out + (str(empty) if empty else ""))
    return "/".join(rows)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="also the first image number: "
                    "runs with different seeds can write into one folder")
    ap.add_argument("--positions", help="positions.json from build_dataset.py merge")
    ap.add_argument("--samples", type=int, default=24, help="render samples (quality against time)")
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--preview", action="store_true", help="also write the labels drawn on a small copy")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    for sub in ("pieces/images", "pieces/labels", "corners/images", "corners/labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    placements = [START]
    if args.positions:
        rows = json.loads(Path(args.positions).read_text())
        placements = sorted({r["placement"] for r in rows if r.get("split") == "train"})

    for i in range(args.count):
        number = args.seed * 100000 + i
        rng = random.Random(number)
        stem = f"render_{number:08d}"
        if (out / "pieces" / "labels" / f"{stem}.txt").exists():
            continue
        r = rng.random()
        placement = (made_up(rng) if r < 0.25 or len(placements) < 2 and r < 0.8
                     else thin(rng.choice(placements), rng))
        L = args.long_side
        size = rng.choice([(L * 3 // 4, L), (L * 3 // 4, L), (L, L * 3 // 4), (L, L), (L * 9 // 16, L), (L, L * 9 // 16)])
        scene, pieces, border = build_scene(rng, placement)
        cam, view = place_camera(scene, rng, border, size)
        corners, boxes = labels(scene, cam, pieces)

        engines = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
        scene.render.engine = "BLENDER_EEVEE" if "BLENDER_EEVEE" in engines else "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = args.samples
        if hasattr(scene.eevee, "use_raytracing"):
            scene.eevee.use_raytracing = rng.random() < 0.5   # reflections on glossy floors
        scene.view_settings.view_transform = rng.choice(["Standard", "AgX", "Filmic"])
        scene.view_settings.exposure = rng.uniform(-1.2, 0.2)
        scene.view_settings.gamma = rng.uniform(0.85, 1.15)
        scene.render.image_settings.file_format = "JPEG"
        scene.render.image_settings.quality = rng.randint(70, 95)
        image_path = out / "pieces" / "images" / f"{stem}.jpg"
        scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)

        (out / "pieces" / "labels" / f"{stem}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes))
        inside = all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in corners)
        if inside:
            xs, ys = [c[0] for c in corners], [c[1] for c in corners]
            box = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys))
            (out / "corners" / "labels" / f"{stem}.txt").write_text(
                "0 " + " ".join(f"{v:.6f}" for v in box) + " "
                + " ".join(f"{x:.6f} {y:.6f} 2" for x, y in corners) + "\n")
            link = out / "corners" / "images" / f"{stem}.jpg"
            if not link.exists():
                link.symlink_to(image_path)
        with open(out / "positions.jsonl", "a") as f:
            f.write(json.dumps(dict(stem=stem, placement=placement, corners=corners,
                                    corners_inside=inside, size=size, **view)) + "\n")
        if args.preview:
            (out / "preview").mkdir(exist_ok=True)
            (out / "preview" / f"{stem}.json").write_text(json.dumps(dict(
                image=str(image_path), corners=corners,
                boxes=[(PIECE_CLASSES[c], x, y, w, h) for c, x, y, w, h in boxes])))
        print(f"[render_boards] {i + 1}/{args.count} {stem} {placement} {view}", flush=True)


if __name__ == "__main__":
    main()
