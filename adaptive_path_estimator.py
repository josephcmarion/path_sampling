import numpy as np
import GPy
from networkx import DiGraph
from networkx.algorithms.shortest_paths.generic import shortest_path


class PathEstimator:

    def __init__(self, potential, dimension, grid_mins, grid_maxs, grid_types):
        """ Class that does map estimation and path sampling. Not documented because I'm not sure what it does yet.

        parameters
        ----------
        potential: function
            has inputs samples, params. Computes the potential at each sample.
            See documentation for self._potential for more details
        dimension: int
            dimension of the path space
        grid_mins: array-like
            the minimum parameter value in each of the Q dimensions
        grid_maxs: array-like
            the maximum parameter value in each of the Q dimensions
        grid_types: list of str
            the type of spacing to using in each path dimension. Must be either 'linear', 'log_increasing', or
            'log_decreasing'


        """
        self._potential = potential
        self.dimension = dimension

        # arguments that have to do with grids
        self.grid_mins = grid_mins
        self.grid_maxs = grid_maxs
        self.grid_types = grid_types
        self.good_types = ['linear', 'log_increasing', 'log_decreasing']
        self.log_increasing_c = 0.01
        self._check_grid_type()  # raises an error if improper types have been specified

        # place holder stuff for the GP regression
        self.pca = None
        self.scaler = None
        self.gps = []
        self.dependent = False
        self.log_transforms = []

        # placeholder for the graph attributes
        self.graph = None
        self.offset = 0

    def _potential(self, samples, params):
        """ a place holder function for the potential, to be specified by the user.
        This function should have the following structure

        parameters
        ----------
        samples: np.array
            K x N x Q array of samples to be used in estimation
        params: np.array
            K x dimension array of path sampling coordinates

        returns
        -------
        np.array
            K x N x dimension array orray of potentials
        """

    def _decompose(self, energies):
        """ takes a matrix of energies and transforms them into the decomposition for modeling

        parameters
        ----------
        energies: np.array
            K x dimension x dimension array of estimated energies at various design point locations

        returns
        -------
        np.array
            K x dimension*(dimension+1) array of decomposed energy matrices
        """

        decomposition = np.linalg.cholesky(energies)

        decomposition[:, range(self.dimension), range(self.dimension)] = \
            np.log(decomposition[:, range(self.dimension), range(self.dimension)])  # log transform the diagonals

        return np.array(map(lambda x: x[np.tril_indices(2)], decomposition))  # flatten into a matrix for GP modeling

    @staticmethod
    def _transform_design(design, log_transforms=None):
        """ transforms columns the design. Currently only does the log transform

        parameters
        ----------
        design: np.array
            M x dimension array of path sampling coordinates
        log_transforms: np.array of boolean
            indicates which columns to log-transforms. If None, no transforms are made

        returns
        -------
        np.array
            transformed design matrix

        """

        # Do nothing if no transforms specified
        if log_transforms is None:
            return design

        # Transform the specified columns
        trans_design = design.copy()
        for c, transform in enumerate(log_transforms):
            if transform:
                trans_design[:, c] = np.log(design[:, c])

        return trans_design

    def _fit_energy(self, samples, design,  log_transforms=None, raw_samples=True, verbose=True):
        """ fits gaussian process to estimate component energy at different parameter configurations.
        Uses samples and paths from various runs of self.sampling

        parameters
        ----------
        samples: np.array
            M x N x Q array of samples to be used in estimation
        design: np.array
            M x dimension array of path sampling coordinates
        log_transforms: np.array of boolean
            indicates which columns to log-transforms. If None, no transforms are made
        raw_samples: bool
            if true, indicates that the samples need to be transformed into potentials
        verbose: bool
            if true prints messages at the beginning of each optimization routine
        """

        self.gps = []  # this will override previously fit gps
        self.log_transforms = log_transforms

        # Turn samples into elements of the cholesky
        if raw_samples:
            potentials = self._potential(samples, design)
        else:
            potentials = samples.copy()

        energies = np.array(map(lambda x: np.cov(x.T), potentials))
        ys = self._decompose(energies)

        # now fit the gps
        transformed_design = self._transform_design(design, self.log_transforms)
        for q in range(ys.shape[1]):
            if verbose:
                print '\nFitting GP #{}'.format(q)
            kernel = GPy.kern.RBF(input_dim=self.dimension)
            model = GPy.models.GPRegression(transformed_design, ys[:, q].reshape(-1, 1), kernel)
            model.optimize_restarts(num_restarts=10, verbose=verbose)
            self.gps.append(model.copy())

    def predict_energy(self, params):
        """ predicts component-wise energy at various parameter locations using trained gps

        parameters
        ----------
        params: np.array
            (M, dimension) array of path parameters. Each row should correspond to a path parameter for the model

        """
        self._check_gps_fit()

        # predict
        transformed_params = self._transform_design(params, self.log_transforms)
        predictions = np.array([gp.predict(transformed_params)[0].flatten() for gp in self.gps]).T

        # transform from Cholesky to expected space
        # this is terrible but it's the best I could find
        lower = np.zeros((params.shape[0], self.dimension, self.dimension))
        for n in range(params.shape[0]):
            lower[n][np.tril_indices(self.dimension)] = predictions[n]

        # exponentiation the diagonal
        lower[:, range(self.dimension), range(self.dimension)] = \
            np.exp(lower[:, range(self.dimension), range(self.dimension)])

        return np.array(map(lambda x: np.matmul(x, x.T), lower))  # I tried very hard to do this with einsum...

    @staticmethod
    def _estimate_step_energy(start_params, stop_params, start_energy, stop_energy):
        """ estimates the total energy used to move one set of params to another using a single step of quadriture

        parameters
        ----------
        start_params: array-like
            (Q, ) array of parameter values at the initial point
        start_params: array-like
            (Q, ) array of parameter values at the end point
        start_energy: np.array
            (Q, ) array of energy values at the initial point
        stop_energy: np.array
            (Q, ) array of energy values at the end point

        returns
        -------
        float:
            the energy required to make the transition from start_params to stop_params
            in statistical terms, this is the variance of the estimator for this step of quadriture
        """

        # compute the change in step size along each parameter
        delta = np.array(start_params) - np.array(stop_params)

        return np.dot(np.dot(delta.T, start_energy+stop_energy), delta)  # average riemann metric

    def _get_grids(self, grid_sizes):
        """ creates custom grids that are specified during initialization. These are used for fitting the energy map,
        establishing the graph, determining the shortest path, and plotting functions

        parameters
        ----------
        grid_sizes: list of int
            the size of the grid along each dimension

        returns
        -------
        list of np.array
            pre-specified grid in each dimension of length n
        """

        grids = []
        for size, grid_min, grid_max, grid_type in zip(grid_sizes, self.grid_mins, self.grid_maxs, self.grid_types):

            if grid_type == 'linear':
                grids.append(np.linspace(grid_min, grid_max, size))

            elif grid_type == 'log_decreasing':
                grids.append(np.exp(np.linspace(np.log(grid_min), np.log(grid_max), size)))

                # grids.append(log_linear_spacing(grid_min, grid_max, size, self.log_increasing_c, True))
            elif grid_type == 'log_increasing':
                grids.append(log_linear_spacing(grid_min, grid_max, size, self.log_increasing_c))

        return grids

    @staticmethod
    def _generate_edge_list(indices, directions, max_strides):
        """ creates an array of nodes and corresponding edges for a directed graph structure specified by directions
        and max_strides. This graph is the used for path learning. Can create graphs for paths of arbitrary dimension.

        parameters
        ----------
        indices: list of np.array
            length Q list where each element corresponds to one dimension. each element should be an array of integers
            from 0 to the total number of elements in that direction
        directions: list of str
            length Q list describing the direction of possible connections for each dimension. Each element should
            be either 'forward' (only moves to larger indices), 'backward' (only moves to smaller indices) or
            'both' can move either forwards or backwards
        max_strides: list of int
            length Q list describing the maximmum number of steps allowed in each dimension

        returns
        -------
        np.array
            (#, Q) array of node indices
        list of np.array
            length # list of np.arrays. The i'th element of this list is an array containing the nodes that are
            connected to the i'th element of the first array. This is terribly written.
        """

        # creates the list of all nodes
        nodes = np.array(np.meshgrid(*indices)).reshape(len(indices), -1).T
        edges_list = []
        lengths = [len(index) for index in indices]

        # create an edge_list for each node
        for node in nodes:
            # create list of possible moves
            edges = []
            for i, (index, length, direction, stride) in enumerate(zip(indices, lengths, directions, max_strides)):

                # determine how far to move
                # the 'default' here is the both option
                min_index = max(0, node[i] - stride)
                max_index = min(length, node[i] + stride + 1)

                if direction == 'both':
                    pass
                elif direction == 'forward':
                    min_index = max(0, node[i] + 1)
                elif direction == 'backward':
                    max_index = min(length, node[i])
                else:
                    raise ValueError('''Error in the {}-th direction.
                        Must be 'forward', 'backward' or 'both' '''.format(i))

                edges.append(range(min_index, max_index))
            edges_list.append(np.array(np.meshgrid(*edges)).reshape(len(indices), -1).T)
        return nodes, edges_list

    def set_offset(self, new_offset):
        """ changes the offset added to the cost at each edge without recreating the graph.
        Useful for testing the effect of different offset values without recreating the graph.

        parameters
        ----------
        new_offset: float
            cost added to the energy at each edge
        """

        self._check_graph_generated()

        for edge in self.graph.edges_iter():
            self.graph[edge[0]][edge[1]]['cost'] += new_offset - self.offset
        self.offset = new_offset

    def shortest_path(self, start, stop, n_interpolate):
        """ creates a path by linearly interpolating additional steps within the optimal path

        parameters
        ----------
        start: tuple
            params corresponding to the starting distribution
        stop: tuple
            params corresponding to the final distribution
        n_interpolate: int
            number of points to interpolate at each step. steps=1 returns the best path

        returns
        -------
        list of tuples
            params for the optimal path using the uniform step rule.
        """

        self._check_graph_generated()
        short_path = shortest_path(self.graph, start, stop, weight='cost')
        path = [short_path[0]]

        for start, stop in zip(short_path[:-1], short_path[1:]):
            params = []
            for q in range(self.dimension):
                params.append(np.linspace(start[q], stop[q], n_interpolate + 1)[1:])

            params = unzip(params)  # change the index to steps
            params = map(tuple, params)  # change lists to tuples
            path += params  # save the changes

        return np.array(path)

    def _average_path_cost(self, path):
        """ returns the average path cost, used to compare paths (for optimizing over the offset)

        parameters
        ----------
        path:
            (M, dimension) array of path parameters

        returns
        -------
        float
            average variance of each step
        """

        energy = self.predict_energy(path)
        cost = np.mean([
            self._estimate_step_energy(start, stop, start_energy, stop_energy) for
            start, stop, start_energy, stop_energy in
            zip(path[:-1], path[1:], energy[:-1], energy[1:])])

        return cost

    def generate_weighted_graph(self, grid_sizes, directions, max_strides, offset=0.1):
        """ creates a network x graph that can be used to find the lowest energy path from one point to another.
        This graph becomes the attribute self.graph

        parameters
        ----------
        grid_sizes: list of int
            the size of the grid along each dimension
        directions: list of str
            length 'dimension' list describing the direction of possible connections for each dimension. Each element should
            be either 'forward' (only moves to larger indices), 'backward' (only moves to smaller indices) or
            'both' can move either forwards or backwards
        max_strides: list of int
            length 'dimension' list describing the maximum number of steps allowed in each dimension
        offset: float
            additional cost added to each step. prevents the path from taking really small steps
        """

        self._check_gps_fit()

        # init some things
        self.offset = offset
        grids = self._get_grids(grid_sizes)
        indices = [range(size) for size in grid_sizes]
        params = np.array(np.meshgrid(*grids)).reshape(self.dimension, -1).T
        predicted_energy = self.predict_energy(params)

        # generate the edge list
        nodes, edges_list = self._generate_edge_list(indices, directions, max_strides)

        # create terrible maps, hope to find a better solution at some point
        # also, convert everything to be a tuple for so it can be used in the graph package
        nodes = [tuple(node) for node in nodes]
        edges_list = [map(tuple, edges) for edges in edges_list]
        node_to_energy = {node: energy for node, energy in zip(nodes, predicted_energy)}
        node_to_params = {node: tuple(params) for node, params in zip(nodes, params)}

        # add elements to graph
        self.graph = DiGraph()
        for node, edges in zip(nodes, edges_list):
            for edge in edges:

                u, v = node_to_params[node], node_to_params[edge]
                cost_u, cost_v = node_to_energy[node], node_to_energy[edge]

                cost = self._estimate_step_energy(u, v, cost_u, cost_v)
                cost += offset

                self.graph.add_edge(u, v, {'cost': cost})

    def estimate_lambda(self, samples, path):
        """ estimates lambda, the log ratio of normalizing constants between the initial distribution
         and the target distribution using a specified path. Uses the thermodynamic integration/path sampling
         method described in Gelman and Meng (1998)

        parameters
        ----------
        samples: np.array
            K x N x Q array of samples to be used in estimation
        path: np.array
            K x dimension array of path sampling coordinates

        returns
        -------
        float:
            the estimated log ratio of normalizing constants between the initial and
            the target distributions
        """

        potentials = self._potential(samples, path)
        U_s = potentials.mean(1)
        deltas = path[1:] - path[:-1]

        return (deltas*(U_s[1:] + U_s[:-1])/2.0).sum()

    def _check_graph_generated(self):
        """ throws an error if the graph hasn't been generated """
        if self.graph is None:
            raise RuntimeError('''Graph has not been generated. Run {}.generate_weighted_graph
                before running this function.'''.format(self.__class__.__name__))

    def _check_gps_fit(self):
        """ raises an error if the gps haven't been fit """
        # make sure the gps have been fit
        if len(self.gps) == 0:
            raise RuntimeError('''GPS muse be trained before running this function. Run
             {}.fit_energy'''.format(self.__class__.__name__))

    def _check_grid_type(self):
        """ raises an error if the grid types are misspecified """
        for grid in self.grid_types:
            if grid not in self.good_types:
                error = 'Invalid grid_type. Must be one of' + \
                        ' '.join([''' '{}','''] * len(self.good_types)).format(*self.good_types) + \
                        'Check initialization of {}'.format(self.__class__.__name__)
                raise ValueError(error)


def log_linear_spacing(grid_min, grid_max, size, c, decreasing=False):
    """ Creates a log linear grid from grid_min to grix_max. If increasing has a positive slope.

    parameters
    ----------
    grid_min: float
        lowest value of the grid
    grid_max: float
        largest value of the grid
    size: int
        number of steps to make
    c: float
        slope of the line
    decreasing: bool
        if true uses an increasing slope

    returns
    -------
    np.array
        log-linearly spaced grid from grid_min to grid_max

    """

    # grid spacing on the log scale
    xs = np.linspace(0, 1, size)
    if decreasing:
        xs = xs[::-1]

    grid = (np.log(xs + c) - np.log(c) + grid_min)

    # rescale to (0,1)
    grid -= grid.min()
    grid /= grid.max()

    # rescale to (min, max)
    grid = grid * (grid_max - grid_min) + grid_min
    if decreasing:
        grid = np.abs(grid-1.0)

    return grid


def unzip(zipped):
    """ converts a list of tuples to a list of lists, the opposite of the zip function

    parameters
    ----------
    zipped: list of array-like
        usually created by applying zip or by using map on a function that has multiple outputs

    returns
    -------
    list of lists
        the first list is all the 1st elements from the tuples, the second lists is the second element from
        each tuple, so on and so forth
    """
    return [map(lambda x: x[i], zipped) for i in range(len(zipped[0]))]
