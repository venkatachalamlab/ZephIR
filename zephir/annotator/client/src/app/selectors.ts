import { State_t } from './model'

export const get_annotation_window_state = (app_state: State_t) =>
    app_state.annotation_window

export const get_annotations_state = (app_state: State_t) =>
    app_state.annotations

export const get_worldlines_state = (app_state: State_t) =>
    app_state.worldlines

export const get_provenances_state = (app_state: State_t) =>
    app_state.provenances

export const get_dataset = (app_state: State_t) =>
    app_state.dataset
