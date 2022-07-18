import React from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux'
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom'
import { configureStore, getDefaultMiddleware } from "@reduxjs/toolkit"
import createSagaMiddleware from 'redux-saga'

import { createMuiTheme, ThemeProvider } from '@material-ui/core/styles'
import { blue, pink } from '@material-ui/core/colors';

import './index.css';
import { App, AppTest, reducer, saga } from './app';
import * as serviceWorker from './serviceWorker';
import { saga_actions } from './app'

const sagaMiddleware = createSagaMiddleware();
const middleware = [...getDefaultMiddleware({ thunk: false }), sagaMiddleware];


const store = configureStore({
  reducer,
  middleware,
});

export type AppDispatch_t = typeof store.dispatch

sagaMiddleware.run(saga);

const darkTheme = createMuiTheme({
  palette: {
    type: 'dark',
    primary: blue,
    secondary: pink,
  },
  typography: {
    fontFamily: [
      'source-code-pro',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
      '"Apple Color Emoji"',
      '"Segoe UI Emoji"',
      '"Segoe UI Symbol"',
    ].join(','),
  },
})

ReactDOM.render(
  <Provider store={store}>
    <ThemeProvider theme={darkTheme}>
      <Router>
        <Switch>
          <Route exact path="/" component={App} />
          <Route path="/AppTest" component={AppTest} />
        </Switch>
      </Router>
    </ThemeProvider>
  </Provider>,
  document.getElementById('root')
)

store.dispatch(saga_actions.annotation_window.fetch_state())
store.dispatch(saga_actions.worldlines.get_worldlines())
store.dispatch(saga_actions.annotations.get_annotations())
store.dispatch(saga_actions.provenances.fetch_provenances())

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
