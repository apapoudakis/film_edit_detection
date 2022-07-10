import multiprocessing


def pool_of_workers(number_of_workers, function, list_of_arguments):
    """
    Parallel execution of a task
    :param number_of_workers: the number of cores to use
    :param function: the function to parallelize
    :param list_of_arguments: arguments of the function
    :return: list with the results of the workers
    """

    pool = multiprocessing.Pool(number_of_workers)

    outputs_async = pool.map_async(function, list_of_arguments)

    output = outputs_async.get()

    return output
