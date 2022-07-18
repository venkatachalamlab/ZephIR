# Flask server to provide read access to image volumes and read/write access
# to annotations stored in a provided path.
#
# This file can be run in two ways:
#  1. For development, use flask run annotator.py
#  2. For production, use annotator [OPTIONS]
"""
This provides a webpage to label key-points in a given dataset.

Usage:
    annotator [options]

Options:
    -h --help                   Show this help.
    -v --version                Show version information.
    --dataset=PATH              Location of data for labeling.
    --port=PORT                 Port to serve on. [default: 5000]
"""

from functools import lru_cache
import io
import os
from pathlib import Path
import subprocess
import traceback
from typing import Tuple

from docopt import docopt
from flask import (Flask, jsonify, send_file, request, send_from_directory)
from gevent.pywsgi import WSGIServer
import numpy as np
from skimage.measure import block_reduce

from .data import (get_slice, get_metadata, Annotation,
                   Worldline, load_annotations, save_annotations,
                   stash_annotations, mip_x, mip_y, mip_z)

from . import rpc as rpc
from ..__version__ import __version__


app = Flask(__name__)
app.logger.setLevel("INFO")

app.dataset = Path(r"D:\data\201124")

(app.annotations, app.worldlines) = load_annotations(app.dataset)


def get_volume(t: int) -> np.ndarray:
    return get_slice(app.dataset, t)


@lru_cache(maxsize=100)
def get_rgba_volume(t: int) -> np.ndarray:
    """Add blank blue and alpha channels to an array with max-stride color
    dimension, permute the color dimension to the min-stride position, and
    rearrange memory to make the color dimension min-stride (last)."""

    raw_vol = get_volume(t)

    if raw_vol.ndim == 4:

        shape_c = raw_vol.shape[0]
        shape_zyx = raw_vol.shape[1:]

        if shape_c == 1:
            g = np.zeros((1, *shape_zyx), dtype=np.uint8)
            b = np.zeros((1, *shape_zyx), dtype=np.uint8)
            rgb = np.concatenate((raw_vol, g, b))

        elif shape_c == 2:
            b = np.zeros((1, *shape_zyx), dtype=np.uint8)
            rgb = np.concatenate((raw_vol, b))

        else:
            rgb = raw_vol[:3, ...]

    else:
        r = np.expand_dims(raw_vol, 0)
        shape_zyx = r.shape[1:]
        g = np.zeros((1, *shape_zyx), dtype=np.uint8)
        b = np.zeros((1, *shape_zyx), dtype=np.uint8)
        rgb = np.concatenate((r, g, b))

    alpha = 255 * np.ones((1, *shape_zyx), dtype=np.uint8)

    rgba_czyx = np.concatenate((rgb, alpha))

    rgba_zyxc = rgba_czyx.transpose((1, 2, 3, 0)).copy()

    return rgba_zyxc


# Define the 2D views of our volume to paint on a screen. HTML canvas expects
# a 3D array (yxc), where the "c" is 4 bytes (rgb-alpha) with the shortest
# stride. These should just return the bytes to draw on a canvas.


def get_z(t: int, idx: int) -> bytes:
    zyxc = get_rgba_volume(t)
    yxc = zyxc[idx]
    return yxc.tobytes()


def get_y(t: int, idx: int) -> bytes:
    zyxc = get_rgba_volume(t)
    zxc = zyxc[:, idx, :, :]
    return zxc.tobytes()


def get_x(t: int, idx: int) -> bytes:
    zyxc = get_rgba_volume(t)
    zyc = zyxc[:, :, idx, :]
    yzc = zyc.transpose((1, 0, 2))
    return yzc.tobytes()


def get_mip_z(t: int, idx: int) -> bytes:
    return np.ascontiguousarray(mip_z(get_rgba_volume(t)))


def get_mip_y(t: int, idx: int) -> bytes:
    return np.ascontiguousarray(mip_y(get_rgba_volume(t)))


def get_mip_x(t: int, idx: int) -> bytes:
    return np.ascontiguousarray(mip_x(get_rgba_volume(t)))


def downsample(v: np.ndarray, scale: Tuple[int, ...]) -> np.ndarray:
    """Downsample a volume by a factor that divides the given dimension."""
    #mean8 = lambda x, **kwargs: np.mean(x, dtype=v.dtype, **kwargs)
    return block_reduce(v, scale, np.max)


def get_vol(t: int, idx: int) -> bytes:
    zyxc = get_rgba_volume(t)
    return downsample(zyxc, (1, idx, idx, 1)).tobytes()


def get_gvol(t: int, idx: int) -> bytes:
    """Just the green channel as an 8-bit array."""
    zyxc = get_rgba_volume(t)
    zyxG = zyxc[:, :, :, 1]
    zyxG_scaled = downsample(zyxG, (1, idx, idx))
    #zyxG_scaled = resize(zyxG_scaled, (1, idx, idx))
    return zyxG_scaled.tobytes()


# Serve the static client files


@app.route('/', methods=['GET'])
def serve_dir_directory_index():
    return send_from_directory("client_build", "index.html")


@app.route('/favicon.ico', methods=['GET'])
def serve_favicon():
    return send_from_directory("client_build", "favicon.ico")


@app.route('/manifest.json', methods=['GET'])
def serve_manifest():
    return send_from_directory("client_build", "manifest.json")


@app.route('/static/js/<path:path>', methods=['GET'])
def serve_static_js(path):
    return send_from_directory(os.path.join("client_build", "static", "js"),
                               path)


@app.route('/static/css/<path:path>', methods=['GET'])
def serve_static_css(path):
    return send_from_directory(os.path.join("client_build", "static", "css"),
                               path)


@app.route('/webgl_colormaps/<path:path>', methods=['GET'])
def serve_colormaps(path):
    return send_from_directory(os.path.join("client_build", "webgl_colormaps"),
                               path)


# End static files.


@app.route("/metadata")
def get_app_metadata():
    metadata = get_metadata(app.dataset)
    return jsonify(metadata)


@app.route("/t/<t>/data/<view>/<view_idx>")
def get_binary_image(t, view, view_idx):

    t = int(t)
    view_fn = "get_{}".format(view)
    view_idx = int(view_idx)

    bytes_to_send = globals()[view_fn](t, view_idx)
    buffer = io.BytesIO(bytes_to_send)

    return send_file(buffer,
                     as_attachment=True,
                     attachment_filename="data.bin",
                     mimetype="application/octet-stream")


@app.route("/t/<t>/annotations", methods=["GET"])
def get_annotations(t):

    t = int(t)
    annotations_for_t = app.annotations.get_t(t).to_jsonable_dict()

    return jsonify(annotations_for_t)


@app.route("/annotations", methods=["POST"])
def handle_annotations():

    app.logger.info("Inserting annotation {}".format(request.json))
    a = Annotation(**request.json)
    inserted_a = app.annotations.insert(a)

    return jsonify(inserted_a.to_jsonable_dict())


@app.route("/annotations/<id>", methods=["PUT", "DELETE"])
def handle_annotation(id):

    id = int(id)

    if request.method == "PUT":

        app.logger.info("Updating annotation {} to {}".format(
            id, request.json))

        try:

            updated_a = app.annotations.update(id, request.json)

        except Exception as e:
            app.logger.warn(
                "Update failed. Returning original. Error: {}".format(e))
            updated_a = app.annotations.get(id)

        return jsonify(updated_a.to_jsonable_dict())

    elif request.method == "DELETE":

        try:
            app.logger.info("Deleting annotation {}".format(id))
            deleted_id = app.annotations.delete(id)
            if deleted_id == id:
                return "ok"
            else:
                return "error"

        except:
            return "error"

    else:

        app.logger.info(f"Received invalid request: {request.method}")
        return "error"


@app.route("/worldlines", methods=["GET", "POST"])
def handle_worldlines():

    if request.method == "GET":

        app.logger.info("Fetching worldlines.")
        return jsonify(app.worldlines.to_jsonable_dict())

    elif request.method == "POST":

        app.logger.info("Creating a new worldline.")
        w = app.worldlines.insert(Worldline())
        return jsonify(w.to_jsonable_dict())

    else:

        app.logger.info(f"Received invalid request: {request.method}")
        return "error"


@app.route("/worldlines/<id>", methods=["PUT", "DELETE"])
def handle_worldline(id):

    id = int(id)

    if request.method == "PUT":

        app.logger.info("Updating worldline {}: {}".format(id, request.json))
        w = app.worldlines.update(id, request.json)
        app.logger.info("Updated worldline {}: {}".format(
            id, w.to_jsonable_dict()))
        return jsonify(w.to_jsonable_dict())

    elif request.method == "DELETE":

        app.logger.info("Deleting worldline {}".format(id))
        deleted_id = app.worldlines.delete(id)
        if deleted_id == id:
            return "ok"
        else:
            return "error"

    else:

        app.logger.info(f"Received invalid request: {request.method}")
        return "error"


@app.route("/load", methods=["POST"])
def handle_load():

    app.logger.info("Loading all annotations and worldlines from disk.")
    stash_annotations(app.annotations, app.worldlines, app.dataset)
    (app.annotations, app.worldlines) = load_annotations(app.dataset)

    return jsonify({"path": str(app.dataset)})


@app.route("/save", methods=["POST"])
def handle_save():

    app.logger.info("Saving all annotations and worldlines to disk.")
    save_annotations(app.annotations, app.worldlines, app.dataset)
    return jsonify({"path": str(app.dataset)})


# RPCs can alter the annotations and worldlines states. They are provided with
# the state of the annotation window, the AnnotationTable, and the
# WorldlinesTable. They can return a list of actions to dispatch, which should
# allow the application to get all the updates that were made.


@app.route("/rpc", methods=["GET", "POST"])
def handle_rpc():

    if request.method == "GET":

        listing = rpc.__all__
        app.logger.info("rpc list: {}".format(listing))

        rpc_list = []
        for proc in listing:
            rpc_list.append({
                "name":
                proc,
                "default_args":
                str(rpc.__dict__.get(proc).__dict__.get("default_args", "")),
                "docstring":
                rpc.__dict__.get(proc).__doc__ or ""
            })

        payload = {"status": "OK", "payload": rpc_list}
        return jsonify(payload)

    elif request.method == "POST":

        data = request.json
        app.logger.info("Received rpc request: {}".format(data["method"]))

        proc = getattr(rpc, data["method"], None)

        try:
            actions = proc(app.dataset, app.annotations, app.worldlines,
                           data["state"], data["arg"], app.logger)

            return jsonify({"status": "ok", "callbacks": actions})

        except Exception as e:
            return jsonify({
                "status": "error",
                "exception": str(e),
                "traceback": traceback.format_exc()
            })

    else:

        app.logger.info(f"Received invalid request: {request.method}")
        return "error"


def main():
    """CLI entry point."""

    args = docopt(__doc__, version=f'ZephIR annotator {__version__}')

    if args["--dataset"]:
        app.dataset = Path(args["--dataset"])
        (app.annotations, app.worldlines) = load_annotations(app.dataset)

    port = int(args["--port"])

    print("Starting a server on port {}".format(port))

    http_server = WSGIServer(("0.0.0.0", port), app)
    http_server.serve_forever()


if __name__ == "__main__":
    main()
