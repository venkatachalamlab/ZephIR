
import React from 'react'

type SimpleFormProps = {
  name: string,
  action: Function
}

type ActionFormProps = SimpleFormProps & { input_parser: Function }

type ActionFormState = { value: string }

export class ActionForm extends
  React.Component<ActionFormProps, ActionFormState>
{

  constructor(props: ActionFormProps) {
    super(props);
    this.state = { value: "0" }

    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleChange(event: any) {
    this.setState({ value: event.target.value });
  }

  handleSubmit(event: any) {
    this.props.action(this.props.input_parser(this.state.value))
    event.preventDefault()
  }

  render() {
    return (
      <form onSubmit={this.handleSubmit}>
        <input type="submit" value={this.props.name} />
        <input type="text"
          defaultValue={this.state.value}
          onChange={this.handleChange} />
      </form>
    );
  }
}

export const FloatForm = (props: SimpleFormProps) =>
  <ActionForm {...props} input_parser={parseFloat} />