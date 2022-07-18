import { createSelector } from 'reselect'

import * as app_selectors from '../app/selectors'

export const get_state = app_selectors.get_worldlines_state

export const get_worldlines = createSelector(
  get_state,
  (s) => s.worldlines
)

export const get_visible_worldlines = createSelector(
  get_worldlines,
  (s) => Object.entries(s)
    .filter((w) => w[1].visible)
    .map((w) => Number(w[0]))
)
