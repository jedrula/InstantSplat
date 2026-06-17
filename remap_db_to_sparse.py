#!/usr/bin/env python3
"""
remap_db_to_sparse.py — Rewrite a COLMAP database so image IDs match a
GLOMAP-output sparse model (images.txt).

GLOMAP's global_mapper reassigns image IDs in its output, so the sparse/0/
images.txt IDs won't match the original feature-extraction database IDs.
colmap image_registrator crashes on this mismatch.

Usage:
    python remap_db_to_sparse.py <database.db> <sparse/0/images.txt> <output.db>

Schema handling:
  GLOMAP-built DBs have an extended schema: `type NOT NULL` in `descriptors`
  plus rigs/frames/pose_priors tables, which must be preserved exactly so
  COLMAP 4.0.4's DatabaseCache::Load doesn't throw. We copy2 the full DB and
  remap IDs in-place. Standard COLMAP-schema DBs get a fresh clean copy.
"""
import shutil
import sqlite3
import sys
from pathlib import Path


def pair_id(id1: int, id2: int) -> int:
    if id1 > id2:
        id1, id2 = id2, id1
    return id1 * 2147483647 + id2


def decode_pair_id(pid: int):
    id2 = pid % 2147483647
    id1 = pid // 2147483647
    return id1, id2


def main():
    if len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} <database.db> <images.txt> <output.db>")

    db_path    = Path(sys.argv[1])
    images_txt = Path(sys.argv[2])
    out_path   = Path(sys.argv[3])

    # ── Read sparse model: name → sparse_id ──────────────────────────────────
    sparse_name_to_id = {}
    with open(images_txt) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        sparse_name_to_id[parts[9]] = int(parts[0])
    print(f"Sparse model: {len(sparse_name_to_id)} images")

    # ── Read database image IDs ───────────────────────────────────────────────
    src = sqlite3.connect(db_path)
    db_rows = src.execute("SELECT image_id, name FROM images").fetchall()
    db_name_to_id = {name: img_id for img_id, name in db_rows}

    # ── Detect schema ─────────────────────────────────────────────────────────
    src_desc_cols = [r[1] for r in src.execute("PRAGMA table_info(descriptors)").fetchall()]
    has_type_col = "type" in src_desc_cols
    src.close()

    # ── Build remap: old_db_id → new_id ──────────────────────────────────────
    remap = {}
    used_ids = set(sparse_name_to_id.values())
    for name, db_id in db_name_to_id.items():
        if name in sparse_name_to_id:
            remap[db_id] = sparse_name_to_id[name]
        else:
            new_id = db_id
            while new_id in used_ids:
                new_id += 100000
            remap[db_id] = new_id
            used_ids.add(new_id)

    changed = sum(1 for o, n in remap.items() if o != n)
    print(f"Remapping {changed}/{len(remap)} image IDs")

    if has_type_col:
        _remap_glomap(db_path, out_path, remap, changed)
    else:
        _remap_colmap_standard(db_path, out_path, remap)


def _remap_glomap(db_path, out_path, remap, changed):
    """
    GLOMAP schema (type col in descriptors, rigs/frames/pose_priors tables).
    copy2 the full DB — preserves all GLOMAP-specific table content so that
    COLMAP 4.0.4 DatabaseCache::Load doesn't throw on inconsistent state.
    Then remap IDs in-place using a two-phase approach (via temp IDs) to avoid
    primary key collisions.
    """
    shutil.copy2(db_path, out_path)
    dst = sqlite3.connect(out_path)
    dst.execute("PRAGMA foreign_keys=OFF")
    dst.execute("PRAGMA journal_mode=WAL")

    needs_remap = {o: n for o, n in remap.items() if o != n}
    if not needs_remap:
        dst.commit()
        dst.close()
        print("  remapped images")
        print("  remapped keypoints")
        print("  remapped descriptors")
        print("  remapped matches")
        print("  remapped two_view_geometries")
        print(f"Done → {out_path}  [GLOMAP schema, no ID changes]")
        return

    # Phase 1: move conflicting old IDs to temp space (old_id + 2B offset)
    OFFSET = 2_000_000_000
    for old_id in needs_remap:
        tmp_id = old_id + OFFSET
        for tbl in ("images", "keypoints", "descriptors"):
            dst.execute(f"UPDATE OR IGNORE {tbl} SET image_id=? WHERE image_id=?",
                        (tmp_id, old_id))
    # Remap pair_ids to temp space
    _remap_pairs_inplace(dst, {o: o + OFFSET for o in needs_remap if o != remap[o]})

    # Phase 2: move from temp space to final new IDs
    for old_id, new_id in needs_remap.items():
        tmp_id = old_id + OFFSET
        for tbl in ("images", "keypoints", "descriptors"):
            dst.execute(f"UPDATE OR IGNORE {tbl} SET image_id=? WHERE image_id=?",
                        (new_id, tmp_id))
    _remap_pairs_inplace(dst, {o + OFFSET: n for o, n in needs_remap.items()})

    dst.commit()
    dst.close()
    print("  remapped images")
    print("  remapped keypoints")
    print("  remapped descriptors")
    print("  remapped matches")
    print("  remapped two_view_geometries")
    print(f"Done → {out_path}  [GLOMAP schema]")


def _remap_pairs_inplace(dst, id_remap):
    """Update pair_id in matches and two_view_geometries for given id changes."""
    if not id_remap:
        return
    for table in ("matches", "two_view_geometries"):
        try:
            rows = dst.execute(f"SELECT pair_id FROM {table}").fetchall()
        except Exception:
            continue
        for (pid,) in rows:
            id1, id2 = decode_pair_id(pid)
            new_id1 = id_remap.get(id1, id1)
            new_id2 = id_remap.get(id2, id2)
            if new_id1 != id1 or new_id2 != id2:
                new_pid = pair_id(new_id1, new_id2)
                dst.execute(f"UPDATE {table} SET pair_id=? WHERE pair_id=?",
                            (new_pid, pid))


def _remap_colmap_standard(db_path, out_path, remap):
    """
    Standard COLMAP schema (no type col, no glomap tables).
    Build a fresh clean output DB and copy/remap data.
    Used for colmap_sift pods (/usr/bin/colmap 3.9.1).
    """
    if out_path.exists():
        out_path.unlink()
    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(out_path)
    dst.execute("PRAGMA foreign_keys=OFF")
    dst.execute("PRAGMA journal_mode=WAL")

    dst.executescript("""
        CREATE TABLE cameras (
            camera_id           INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
            model               INTEGER                             NOT NULL,
            width               INTEGER                             NOT NULL,
            height              INTEGER                             NOT NULL,
            params              BLOB,
            prior_focal_length  INTEGER                             NOT NULL
        );
        CREATE TABLE images (
            image_id  INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
            name      TEXT                                NOT NULL UNIQUE,
            camera_id INTEGER                             NOT NULL,
            prior_qw  REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
            prior_tx  REAL, prior_ty REAL, prior_tz REAL,
            CONSTRAINT image_id_check CHECK(image_id >= 0 AND image_id < 2147483647),
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
        );
        CREATE TABLE keypoints (
            image_id  INTEGER  PRIMARY KEY  NOT NULL,
            rows      INTEGER               NOT NULL,
            cols      INTEGER               NOT NULL,
            data      BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );
        CREATE TABLE descriptors (
            image_id  INTEGER  PRIMARY KEY  NOT NULL,
            rows      INTEGER               NOT NULL,
            cols      INTEGER               NOT NULL,
            data      BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );
        CREATE TABLE matches (
            pair_id  INTEGER  PRIMARY KEY  NOT NULL,
            rows     INTEGER               NOT NULL,
            cols     INTEGER               NOT NULL,
            data     BLOB
        );
        CREATE TABLE two_view_geometries (
            pair_id  INTEGER  PRIMARY KEY  NOT NULL,
            rows     INTEGER               NOT NULL,
            cols     INTEGER               NOT NULL,
            data     BLOB,
            config   INTEGER               NOT NULL,
            F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB
        );
    """)

    for row in src.execute("SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras").fetchall():
        dst.execute("INSERT INTO cameras VALUES (?,?,?,?,?,?)", row)
    print("  remapped cameras")

    for img_id, name, cam_id in src.execute("SELECT image_id, name, camera_id FROM images").fetchall():
        dst.execute("INSERT OR IGNORE INTO images (image_id, name, camera_id) VALUES (?,?,?)",
                    (remap.get(img_id, img_id), name, cam_id))
    print("  remapped images")

    for img_id, rows, cols, data in src.execute("SELECT image_id, rows, cols, data FROM keypoints").fetchall():
        dst.execute("INSERT OR IGNORE INTO keypoints VALUES (?,?,?,?)",
                    (remap.get(img_id, img_id), rows, cols, data))
    print("  remapped keypoints")

    for img_id, rows, cols, data in src.execute("SELECT image_id, rows, cols, data FROM descriptors").fetchall():
        dst.execute("INSERT OR IGNORE INTO descriptors VALUES (?,?,?,?)",
                    (remap.get(img_id, img_id), rows, cols, data))
    print("  remapped descriptors")

    for table in ("matches", "two_view_geometries"):
        try:
            col_names = [r[1] for r in src.execute(f"PRAGMA table_info({table})").fetchall()]
        except sqlite3.OperationalError:
            continue
        rows = src.execute(f"SELECT * FROM {table}").fetchall()
        placeholders = ",".join("?" * len(col_names))
        pid_idx = col_names.index("pair_id")
        for row in rows:
            row = list(row)
            id1, id2 = decode_pair_id(row[pid_idx])
            row[pid_idx] = pair_id(remap.get(id1, id1), remap.get(id2, id2))
            dst.execute(f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})", row)
        print(f"  remapped {table}")

    dst.commit()
    dst.close()
    src.close()
    print(f"Done → {out_path}  [COLMAP-standard schema]")


if __name__ == "__main__":
    main()
