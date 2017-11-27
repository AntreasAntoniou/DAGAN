def count_parameters(network_variables, name):
    """
    This method counts the total number of unique parameters for a list of variable objects
    :param network_variables: A list of tf network variable objects
    :param name: Name of the network
    """
    total_parameters = 0
    for variable in network_variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value

        total_parameters += variable_parametes
    print(name, total_parameters)


def view_names_of_variables(variables):
    """
    View all variable names in a tf variable list
    :param variables: A list of tf variables
    """
    for variable in variables:
        print(variable)

