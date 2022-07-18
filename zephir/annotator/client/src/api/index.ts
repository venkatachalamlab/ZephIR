import { LRUMap } from '../third_party/lru'
import { AnnotationsState_t, Annotation_t, annotation_id_t } from '../annotations/model'
import { Worldline_t, Worldlines_t, worldline_id_t } from '../worldlines/model'

const image_cache = new LRUMap<string, Uint8ClampedArray>(5000)
const annotation_cache = new LRUMap<number, AnnotationsState_t>(1000)

// Annotation window

export const fetch_metadata = () => {
  return fetch("/metadata")
    .then(x => x.json())
}

export async function fetch_data_cache(
  url: string,
  signal: AbortSignal): Promise<Uint8ClampedArray> {

  if (image_cache.has(url)) {
    const uint8_array = image_cache.get(url)
    if (uint8_array !== undefined) {
      return uint8_array
    }
  }

  return await fetch(url, { signal })
    .then((x) => x.arrayBuffer())
    .then((x) => {
      const array = new Uint8ClampedArray(x);
      image_cache.set(url, array);
      return array;
    })
    .catch(err => {
      if (err.name === "AbortError") {
        // console.log("Fetch aborted:", url)
      }
      else {
        console.error(err)
      }
    }) as Uint8ClampedArray

}

export async function fetch_data(
  url: string,
  signal: AbortSignal): Promise<Uint8ClampedArray> {

  return await fetch(url, { signal })
    .then((x) => x.arrayBuffer())
    .then((x) => new Uint8ClampedArray(x))
    .catch(err => {
      if (err.name === "AbortError") {
        // console.log("Fetch aborted:", url)
      }
      else {
        console.error(err)
      }
    }) as Uint8ClampedArray

}

// Annotations

export type bare_annotation_t = {
  x: number,
  y: number,
  z: number,
  t_idx: number,
  worldline_id: number
}

export const insert_annotation = (annotation: bare_annotation_t) => {
  return fetch("/annotations", {
    method: "POST", // or "PUT"
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(annotation),
  })
    .then((response) => response.json())
}

export function fetch_annotations_cache(t: number):
  AnnotationsState_t | undefined {

  if (annotation_cache.has(t)) {
    const annotations = annotation_cache.get(t)
    if (annotations !== undefined) {
      return annotations
    }
  }

}

// export const fetch_annotations = (t: number) =>{
//   return fetch(`/t/${t}/annotations`)
//   .then((response) => response.json())
// }

export async function fetch_annotations(t: number, signal: AbortSignal):
  Promise<AnnotationsState_t> {

  return await fetch(`/t/${t}/annotations`, { signal })
    .then((x) => x.json())
    .then((x) => {
      annotation_cache.set(t, x);
      return x;
    })
    .catch(err => {
      if (err.name === "AbortError") {
        // console.log("Fetch aborted: Annotations for ", t)
      }
      else {
        console.error(err)
      }
    }) as AnnotationsState_t
}

export const update_annotation = (a: Annotation_t) => {
  return fetch(`/annotations/${a.id}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(a),
  })
    .then((response) => response.json())
}

export const delete_annotation = (id: annotation_id_t) => {
  return fetch(`/annotations/${id}`, { method: "DELETE" })
}

// Worldlines

export const create_worldline = () => {
  return fetch(`/worldlines`, { method: "POST" })
    .then((response) => response.json())
}

export const fetch_worldlines = () => {
  return fetch(`/worldlines`, { method: "GET" })
    .then((response) => (response.json() as Promise<Worldlines_t>))
}

export const update_worldline = (w: Partial<Worldline_t>) => {
  return fetch(`/worldlines/${w.id}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(w),
  })
    .then((response) => response.json())
}

export const delete_worldline = (id: worldline_id_t) => {
  return fetch(`/worldlines/${id}`, { method: "DELETE" })
}

// Save worldlines and annotations

export const save = () => {
  return fetch("/save", {
    method: "POST",
  })
    .then((response) => response.json())
    .then((data) => {
      window.alert(`Save successful: ${data.path}`);
      console.log("Saved to: ", data)
    })
    .catch((error) => {
      window.alert("Save failed. See console.")
      console.error("Error:", error);
    });
}

// Load worldlines and annotations

export const load = () => {
  return fetch("/load", {
    method: "POST",
  })
    .then((response) => response.json())
    .then((data) => {
      window.alert(`Load successful: ${data.path}`);
    })
    .catch((error) => {
      window.alert("Load failed. See console.")
      console.error("Error:", error);
    });
}

// RPC

export const fetch_rpcs = () => {
  return fetch("/rpc")
    .then(x => x.json())
}

export const rpc = (method: string, arg: string, state: AnnotationsState_t) => {

  return fetch("/rpc", {
    method: "POST", // or "PUT"
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ method, arg, state }),
  })
    .then((response) => response.json())

}