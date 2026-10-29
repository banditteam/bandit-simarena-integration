import argparse
import copy
import json
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.spatial.transform import Rotation

def values(s):
    return [float(x) for x in s.split()]

def inertia_properties(body):
    node = body.find("inertial")
    if node is None:
        raise ValueError("Missing inertia: " + body.get("name"))

    mass = float(node.get("mass"))
    center = values(node.get("pos", "0 0 0"))

    if node.get("fullinertia"):
        xx, yy, zz, xy, xz, yz = values(node.get("fullinertia"))
        tensor = np.array([
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz]
        ])

        moments, axes = np.linalg.eigh(tensor)

        if min(moments) <= 0:
            raise ValueError("Invalid inertia: " + body.get("name"))

        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1

        q = Rotation.from_matrix(axes).as_quat()
        quat = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
        inertia = moments.tolist()

    else:
        inertia = values(node.get("diaginertia"))
        quat = values(node.get("quat", "1 0 0 0"))

    return {
        "type": "rigid",
        "motion": "dynamic",
        "mass": mass,
        "centerOfMass": center,
        "inertia": inertia,
        "inertiaQuat": quat
    }

def capsule_quat(direction):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(z, direction))

    if dot < -0.999999:
        return [0.0, 1.0, 0.0, 0.0]

    q = np.r_[1.0 + dot, np.cross(z, direction)]
    q /= np.linalg.norm(q)
    return q.tolist()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", required=True)
    p.add_argument("--visual", required=True)
    p.add_argument("--g1", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    root = ET.parse(args.xml).getroot()

    visual = json.loads(
        Path(args.visual).read_text(encoding="utf-8-sig")
    )
    g1 = json.loads(
        Path(args.g1).read_text(encoding="utf-8-sig")
    )

    assert visual["_format"] == "simarena/asset@1.0"
    assert g1["_format"] == "simarena/asset@1.0"

    result = copy.deepcopy(visual)
    result["name"] = "bandit_rp1_physics_v1"

    opt = root.find("option")
    result["sim"] = {
        "timestep": float(opt.get("timestep", "0.005")),
        "integrator": opt.get("integrator", "implicitfast"),
        "solver": opt.get("solver", "Newton").lower(),
        "iterations": int(opt.get("iterations", "100"))
    }

    bodies = {
        b.get("name"): b
        for b in root.findall(".//body")
    }

    entities = {
        e["name"]: e
        for e in result["entities"]
        if e["name"] in bodies
    }

    motors = {
        m.get("joint"): m
        for m in root.findall("./actuator/motor")
    }

    missing = set(bodies) - set(entities)
    if missing:
        raise ValueError("Missing bodies: " + str(missing))

    if len(bodies) != 25 or len(motors) != 24:
        raise ValueError("Unexpected robot structure")

    joint_count = 0
    collision_count = 0

    for name, body in bodies.items():
        entity = entities[name]
        physics = inertia_properties(body)

        if name == "base_link":
            if body.find("freejoint") is None:
                raise ValueError("Missing root freejoint")

            physics["motion"] = "static"
            entity["joint"] = {"type": "free"}

        else:
            joint = body.find("joint")

            if joint is None or joint.get("type") != "hinge":
                raise ValueError("Invalid joint: " + name)

            motor = motors.get(joint.get("name"))
            if motor is None:
                raise ValueError("Missing motor: " + name)

            entity["joint"] = {
                "type": "hinge",
                "axis": values(joint.get("axis", "0 0 1")),
                "range": values(joint.get("range")),
                "armature": float(joint.get("armature", "0")),
                "drive": {
                    "mode": "torque",
                    "ctrlrange": values(motor.get("ctrlrange"))
                }
            }

            joint_count += 1

        entity["physics"] = physics

        collision_geoms = []

        for geom in body.findall("geom"):
            if geom.get("class") != "collision":
                continue

            if geom.get("type") != "capsule":
                raise ValueError("Unsupported collision geometry")

            radius = values(geom.get("size"))[0]

            if geom.get("fromto"):
                ft = np.asarray(values(geom.get("fromto")))
                a, b = ft[:3], ft[3:]
                direction = b - a

                collision = {
                    "type": "capsule",
                    "radius": radius,
                    "height": float(np.linalg.norm(direction)),
                    "pos": ((a + b) / 2).tolist(),
                    "quat": capsule_quat(direction),
                    "group": "collision",
                    "contype": 1,
                    "conaffinity": 1
                }

            else:
                size = values(geom.get("size"))
                collision = {
                    "type": "capsule",
                    "radius": radius,
                    "height": 2 * size[1] if len(size) > 1 else 0.000001,
                    "pos": values(geom.get("pos", "0 0 0")),
                    "group": "collision",
                    "contype": 1,
                    "conaffinity": 1
                }

            collision_geoms.append(collision)

        if collision_geoms:
            entity.setdefault("geoms", []).extend(collision_geoms)
            entity["collider"] = {"groups": ["collision"]}
            collision_count += len(collision_geoms)

    assert joint_count == 24
    assert collision_count == 41

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8"
    )

    print("PASS: 25 physical bodies")
    print("PASS: 24 torque-driven hinge joints")
    print("PASS: 1 free root")
    print("PASS: 41 capsule collision geometries")
    print("PASS: Visual asset preserved")
    print("OUTPUT:", out)

if __name__ == "__main__":
    main()
