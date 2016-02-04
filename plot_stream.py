""" This file contains a class useful for plotting streamline plots with
steady state information using matplotlib """

from __future__ import division

import warnings

import numpy as np
from numpy.linalg import norm, eig, eigvals, solve
from scipy.optimize import fsolve, newton

import matplotlib.pyplot as plt


class SteadyStateStreamPlot(object):
    """
    class that can analyze a 2D differential equation and calculated and plot
    steady states, stream lines, and similar information.
    """
    
    def __init__(self, func, region, region_constraint=None):
        """
        `func` is the vector function that defines the 2D flow field.
        The function must accept and also return an array with 2 elements
        `region` defines the region of interest for plotting. Four numbers need
            to be specified: [left, bottom, right, top].
        """
        self.func = func
        self.region = region
        if region_constraint is None:
            self.region_constraint = set()
        else:
            self.region_constraint = set(region_constraint)
        
        self.steady_states = None
        self.step = min(
            region[2] - region[0],
            region[3] - region[1]
        )/100 

        # determine region of interest
        self.rect = [
            self.region[0] - 1e-4, self.region[1] - 1e-4,
            self.region[2] + 1e-4, self.region[3] + 1e-4
        ]


    def point_in_region(self, point, strict=False):
        """
        checks whether `point` is in the region of interest
        """
        if strict:
            rect = self.region
        else:
            rect = self.rect
        return rect[0] < point[0] < rect[2] and rect[1] < point[1] < rect[3]


    def point_at_border(self, point, tol=1e-6):
        """
        returns 0 if point is not at a border, otherwise returns a
        positive number indicating which border the points belongs to
        """
        res = 0
        if np.abs(point[0] - self.region[0]) < tol:
            res += 1
        if np.abs(point[1] - self.region[1]) < tol:
            res += 2
        if np.abs(point[0] - self.region[2]) < tol:
            res += 4
        if np.abs(point[1] - self.region[3]) < tol:
            res += 8
        return res


    def jacobian(self, point, tol=1e-6):
        """
        returns the Jacobian around a point
        """
        jacobian = np.zeros((2, 2))
        jacobian[0, :] = self.func(point + np.array([tol, 0]))
        jacobian[1, :] = self.func(point + np.array([0, tol]))
        return jacobian


    def point_is_steady_state(self, point, tol=1e-6):
        """
        Checks whether `point` is a steady state
        """

        vel = self.func(point) #< velocity at the point
        x_check, y_check = False, False #< checked direction
        
        # check special boundary cases
        if 'left' in self.region_constraint and np.abs(point[0] - self.region[0]) < 1e-8:
            if vel[0] > 0:
                return False
            x_check = True 
        elif 'right' in self.region_constraint and np.abs(point[0] - self.region[2]) < 1e-8:
            if vel[0] < 0:
                return False
            x_check = True 

        if 'bottom' in self.region_constraint and np.abs(point[1] - self.region[1]) < 1e-8:
            if vel[1] > 0:
                return False
            y_check = True
        elif 'top' in self.region_constraint and np.abs(point[1] - self.region[3]) < 1e-8:
            if vel[1] < 0:
                return False
            y_check = True
        
        # check the remaining directions
        if x_check and y_check:
            return True # both x and y direction are stable
        elif x_check:
            return np.abs(vel[1]) < tol # y direction has to be tested
        elif y_check:
            return np.abs(vel[0]) < tol # x direction has to be tested
        else:
            return norm(vel) < tol # both directions have to be tested
        

    def point_is_stable(self, point, tol=1e-5):
        """
        returns true if a given steady state is stable
        """
        
        if not self.point_is_steady_state(point, tol):
            raise ValueError('Supplied point is not a steady state')
        
        jacobian = self.jacobian(point, tol)
        x_check, y_check = False, False #< checked direction
        
        # check special boundary cases
        if 'left' in self.region_constraint and np.abs(point[0] - self.region[0]) < 1e-8:
            x_check = True 
        elif 'right' in self.region_constraint and np.abs(point[0] - self.region[2]) < 1e-8:
            x_check = True 

        if 'bottom' in self.region_constraint and np.abs(point[1] - self.region[1]) < 1e-8:
            y_check = True
        elif 'top' in self.region_constraint and np.abs(point[1] - self.region[3]) < 1e-8:
            y_check = True
        
        # check the remaining directions
        if x_check and y_check:
            return True # both x and y direction are stable
        elif x_check:
            return jacobian[1, 1] < 0 # y direction has to be tested
        elif y_check:
            return jacobian[0, 0] < 0 # x direction has to be tested
        else:
            return all(eigvals(jacobian) < 0) # both directions have to be tested


    def get_steady_states_at_x_boundary(self, grid_points=32, loc='lower', points=None):
        """
        finds steady state points along the x-boundary at position `loc`
        """
        
        if points is None:
            points = np.array([[]]) #< array that will contain all the points
            
        if loc == 'lower':
            y0 = self.region[1]
            direction = 1
        elif loc == 'upper':
            y0 = self.region[3]
            direction = -1
        
        xs, dist = np.linspace(self.region[0], self.region[2], grid_points, retstep=True)
        
        # consider a horizontal boundary
        def func1D(x):
            return self.func((x, y0))[0]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for x0 in xs:
                try:
                    x_guess = newton(func1D, x0)
                except (RuntimeError, RuntimeWarning):
                    continue
                
                guess = np.array([x_guess, y0])
                dx, dy = self.func(guess)
                
                if norm(dx) > 1e-5 or direction*dy > 0:
                    continue
                
                if not self.point_in_region(guess):
                    continue
        
                if points.size == 0:
                    points = guess[None, :]
                elif np.all(np.abs(points - guess[None, :]).sum(axis=1) > dist):
                    points = np.vstack((points, guess))
                
        return points


    def get_steady_states_at_y_boundary(self, grid_points=32, loc='left',
                                        points=None):
        """
        finds steady state points along the y-boundary at position `loc`
        """
        
        if points is None:
            points = np.array([[]]) #< array that will contain all the points
            
        if loc == 'left':
            x0 = self.region[0]
            direction = 1
        elif loc == 'right':
            x0 = self.region[2]
            direction = -1
        
        ys, dist = np.linspace(self.region[1], self.region[3], grid_points,
                               retstep=True)
        
        # consider a horizontal boundary
        def func1D(y):
            return self.func((x0, y))[1]
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for y0 in ys:
                try:
                    y_guess = newton(func1D, y0)
                except (RuntimeError, RuntimeWarning):
                    continue

                guess = np.array([x0, y_guess])
                dx, dy = self.func(guess)
                
                if direction*dx > 0 or norm(dy) > 1e-5:
                    continue
                
                if not self.point_in_region(guess):
                    continue
        
                if points.size == 0:
                    points = guess[None, :]
                elif np.all(np.abs(points - guess[None, :]).sum(axis=1) > dist):
                    points = np.vstack((points, guess))
                
        return points
                    

    def get_steady_states(self, grid_points=32):
        """
        determines all steady states in the region.
        `grid_points` is the number of points to take as guesses along each axis.
        `region_constraint` can be a list of identifiers ('left', 'right', 'top', 'bottom')
        indicating that the respective boundary poses a constraint on the dynamics and 
        there may be stationary points along the boundary
        """
        if self.steady_states is None:

            points = np.array([[]]) #< array that will contain all the points
            
            xs, dx = np.linspace(self.region[0], self.region[2], grid_points,
                                 retstep=True)
            ys, dy = np.linspace(self.region[1], self.region[3], grid_points,
                                 retstep=True)
            
            # check the border separately if requested
            if 'left' in self.region_constraint:
                points = self.get_steady_states_at_y_boundary(grid_points,
                                                              'left', points)
                xs = xs[1:] # skip this point in future calculations
            if 'bottom' in self.region_constraint:
                points = self.get_steady_states_at_x_boundary(grid_points,
                                                              'lower', points)
                ys = ys[1:] # skip this point in future calculations
            if 'right' in self.region_constraint:
                points = self.get_steady_states_at_y_boundary(grid_points,
                                                              'right', points)
                xs = xs[:-1] # skip this point in future calculations
            if 'top' in self.region_constraint:
                points = self.get_steady_states_at_x_boundary(grid_points,
                                                              'upper', points)
                ys = ys[:-1] # skip this point in future calculations
                        
            xs, ys = np.meshgrid(xs, ys)
            dist = max(dx, dy)
            
            # find all stationary points
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                for guess in zip(xs.flat, ys.flat):
                    
                    try:
                        guess = fsolve(self.func, guess, xtol=1e-8)
                    except (RuntimeError, RuntimeWarning):
                        continue
                    
                    if norm(self.func(guess)) > 1e-5:
                        continue
            
                    guess = np.array(guess)
            
                    if not self.point_in_region(guess):
                        continue
            
                    if points.size == 0:
                        points = guess[None, :]
                    elif np.all(np.abs(points - guess[None, :]).sum(axis=1)
                                > dist):
                        points = np.vstack((points, guess))
            
            # determine stability of the steady states
            stable, unstable = [], []
            for point in points:
                if self.point_is_stable(point):
                    stable.append(point)
                else:
                    unstable.append(point)
        
            stable = np.array(stable)
            unstable = np.array(unstable)
        
            self.steady_states = (stable.reshape((-1, 2)),
                                  unstable.reshape((-1, 2)))
        
        return self.steady_states
    

    def plot_steady_states(self, ax=None, color='k', **kwargs):
        """
        plots the steady states
        """
        stable, unstable = self.get_steady_states()
        
        if ax is None:
            ax = plt.gca()
        
        if stable.size > 0:
            ax.plot(
                stable[:, 0], stable[:, 1],
                'o', color=color, clip_on=False, **kwargs
            )
        if unstable.size > 0:
            ax.plot(
                unstable[:, 0], unstable[:, 1],
                'o', markeredgecolor=color, markerfacecolor='none', 
                markeredgewidth=1, clip_on=False, **kwargs
            )


    def plot_stationarity_line(self, axis=0, step=None, **kwargs):
        """
        plots the lines along which the variable plotted on `axis` is stationary
        """
        
        if step is None:
            step = self.step
        
        i_vary = 1 - axis #< index to vary
        
        # collect all start points
        points = np.concatenate(self.get_steady_states()).tolist() 
        
        # build vector for right hand side
        eps = np.zeros(2)
        eps[i_vary] = 1e-6
        
        def rhs(angle, point, step):
            """ rhs of the differential equation """
            x = point + step*np.array([np.cos(angle), np.sin(angle)])
            return self.func(x)[axis]
        
        def ensure_trajectory(point, ds):
            """ make sure we are actually on the trajectory """
            angle = np.arctan2(ds[1], ds[0])
            step = norm(ds)
            angle = newton(rhs, x0=angle, args=(point, step))
            return point + step*np.array([np.cos(angle), np.sin(angle)])
                   
        def get_traj(point, direction):
            """ retrieve an array with trajectory points """
            x0, dx = np.array(point), direction
            xs = [x0]
            
            while True:
                x0 = xs[-1]
                dx *= step/norm(dx)

                # check distances to all endpoints
                for p in points:
                    if norm(p - x0 - dx) < step:
                        xs.append(p) # add the point to the line
                        # skip over this point in the integration
                        dx *= 2 # step over the steady state
                        points.remove(p)
                        break
              
                # make sure we're on the trajectory
                try:
                    x1 = ensure_trajectory(x0, dx)
                except (RuntimeError, RuntimeWarning):
                    break

                # check whether trajectory left the system
                if not self.point_in_region(x1, strict=True):
                    break

                xs.append(x1)
                # get step by extrapolating from last step
                dx = x1 - x0
            
            return np.array(xs)
                                
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            while len(points) > 0:
                point = points.pop()
                
                dx = solve(self.jacobian(point), eps)
    
                # follow the trajectory in both directions
                traj = np.concatenate((
                    get_traj(point, -dx)[::-1],
                    get_traj(point, +dx)[1:]
                ))
    
                if len(traj) > 3:
                    plt.plot(traj[:, 0], traj[:, 1], **kwargs)
               
        
    def plot_streamline(self,
            x0, ds=0.01, endpoints=None, point_margin=None,
            ax=None, skip_initial_points=False, color='k', **kwargs
        ):
        """
        Plots a single stream line starting at x0 evolving under the flow.
        `ds` determines the step size (if it is negative we evolve back in time).
        """
        if ax is None:
            ax = plt.gca()

        if endpoints is None:
            endpoints = np.concatenate(self.get_steady_states())
            
        if point_margin is None:
            point_margin = 5*self.step
        
        traj = [np.array(x0)]
        
        while True:
            x = traj[-1] # last point

            # check whether trajectory left the system
            if not self.point_in_region(x):
                break
            
            # check distances to endpoints
            if endpoints.size > 0:
                dist_to_endpoints = np.sqrt(
                    ((endpoints - x[None, :])**2).sum(axis=1)
                ).min()
                if dist_to_endpoints < point_margin:
                    break
            
            # iterate one step
            dx = np.array(self.func(x))
            traj.append(x + ds*dx/norm(dx))

        # finished iterating => plot
        if len(traj) > 1:
            traj = np.array(traj)
            if skip_initial_points:
                i_start = int(point_margin/np.abs(ds))
            else:
                i_start = 0
            plt.plot(traj[i_start::10, 0], traj[i_start::10, 1], '-',
                     color=color, **kwargs)

            # indicate direction with an arrow in the middle
            # TODO: calculate the midpoint based on actual pathlength
            i = int(0.5*len(traj)) #< midpoint
            try:
                dx = np.sign(ds)*(traj[i+5] - traj[i-5])
            except IndexError:
                dx = np.sign(ds)*(traj[i] - traj[i-1])
            dx *= 1e-2/norm(dx)
            plt.arrow(
                traj[i, 0], traj[i, 1], dx[0], dx[1],
                width=self.step/10, color=color, length_includes_head=True,
                zorder=10, clip_on=False
            )


    def plot_streamlines(self, point, angles=None, stable_direction=None,
                         **kwargs):
        """
        Plots streamlines starting from points around `point`.
        `angles` are given in degrees to avoid factors of pi
        """
        point = np.asarray(point)
        if stable_direction is None:
            stable_direction = self.point_is_stable(point)

        stable, unstable = self.get_steady_states()
        if stable_direction:
            ds = -0.01 #< integration step and direction
            endpoints = unstable
        else:
            ds = 0.01 #< integration step and direction
            endpoints = stable
        
        if angles is None:
            angles = np.arange(0, 360, 45)
        else:
            angles = np.asarray(angles)
        
        for angle in angles:
            # initial point: use exact forms to avoid numerical instabilities
            if angle == 0:
                dx = np.array([1, 0])
            elif angle == 90:
                dx = np.array([0, 1])
            elif angle == 180:
                dx = np.array([-1, 0])
            elif angle == 270:
                dx = np.array([0, -1])
            else:
                angle *= np.pi/180
                dx = np.array([np.cos(angle), np.sin(angle)])
                
            x0 = point + np.abs(ds)*dx
            
            self.plot_streamline(x0, ds=ds, endpoints=endpoints,
                                 skip_initial_points=True, **kwargs)

    
    def plot_heteroclinic_orbits(self, **kwargs):
        """
        Plots the heteroclinic orbits connecting different stationary states
        """
        stable, unstable = self.get_steady_states()
        points = np.concatenate((stable, unstable))
        
        # iterate through all steady states that are not border points
        for point in points:
            
#             if self.point_at_border(point):
#                 continue
        
            # determine stable and unstable directions
            eigenvalues, eigenvectors = eig(self.jacobian(point))
            
            # iterate through all eigenvalues
            for k, ev in enumerate(eigenvalues):
                if ev.real < 0:
                    # stable state
                    ds = -0.001
                    endpoints = unstable
                else:
                    # unstable state
                    ds = 0.001
                    endpoints = stable
                
                # start trajectories in both directions
                for dx in (ds, -ds):
                    x0 = point + 1e-6*dx*eigenvectors[:, k]
                    self.plot_streamline(x0, ds, endpoints=endpoints,
                                         skip_initial_points=True, **kwargs)



if __name__ == "__main__":
    print('This file is intended to be used as a module.')
