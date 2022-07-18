import { createSelector } from 'reselect'

import * as app_selectors from '../app/selectors'

export const get_state = app_selectors.get_provenances_state

export const get_visible_provenances = createSelector(
    get_state,
    (x) => {
        console.log(x)
        return Object.entries(x)
        .filter((p) => p[1].visible)
        .map((p) => p[1].id)}
)