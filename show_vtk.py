"""file from https://github.com/toinsson/pyrealsense/blob/master/examples/show_vtk.py

"""

import time
import threading

import numpy as np

import vtk
import vtk.util.numpy_support as vtk_np



class VTKActorWrapper(object):
    def __init__(self, nparray, color=[1,1,1], opacity=1):
        super(VTKActorWrapper, self).__init__()
        self.actors = []
        self.addPoints(nparray, color, opacity)
        self.mean = nparray.mean(axis=0)

    def addPlane(self, w):
        source = vtk.vtkPlaneSource()
        source.SetCenter(self.mean)
        source.SetNormal(w[0], w[1], w[2])
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetScale(0.1)
        self.actors.append(actor)

    def addNormal(self, w):
        arrowSource = vtk.vtkArrowSource()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(arrowSource.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetScale(0.05)

        self.actors.append(actor)

    def addPoints(self, nparray, color, opacity):
        self.nparray = nparray

        nCoords = nparray.shape[0]
        nElem = nparray.shape[1]

        self.verts = vtk.vtkPoints()
        self.cells = vtk.vtkCellArray()
        self.scalars = None

        self.pd = vtk.vtkPolyData()
        self.verts.SetData(vtk_np.numpy_to_vtk(nparray))
        self.cells_npy = np.vstack([np.ones(nCoords, dtype=np.int64),
                                    np.arange(nCoords, dtype=np.int64)]).T.flatten()
        self.cells.SetCells(nCoords, vtk_np.numpy_to_vtkIdTypeArray(self.cells_npy))
        self.pd.SetPoints(self.verts)
        self.pd.SetVerts(self.cells)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputDataObject(self.pd)

        actor = vtk.vtkActor()
        actor.SetMapper(self.mapper)
        actor.GetProperty().SetRepresentationToPoints()
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        self.actors.append(actor)

    def update(self, threadLock, update_on):
        #thread = threading.Thread(target=self.update_actor, args=(threadLock, update_on))
        #thread.start()
        return

    def update_actor(self, threadLock, update_on):
        while (update_on.is_set()):
            time.sleep(0.01)
            threadLock.acquire()
            self.pd.Modified()
            threadLock.release()


class VTKVisualisation(object):
    def __init__(self, threadLock, axis=True,):
        super(VTKVisualisation, self).__init__()

        self.threadLock = threadLock

        self.ren = vtk.vtkRenderer()
        #self.ren.AddActor(actorWrapper.actor)
        #self.addActor(actorWrapper)

        self.axesActor = vtk.vtkAxesActor()
        self.axesActor.AxisLabelsOff()
        self.axesActor.SetTotalLength(0.02, 0.02, 0.02)
        self.ren.AddActor(self.axesActor)

        self.renWin = vtk.vtkRenderWindow()
        self.renWin.SetSize(640, 480)
        self.renWin.AddRenderer(self.ren)

        ## IREN
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.iren.Initialize()

        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(self.style)

        self.iren.AddObserver("TimerEvent", self.update_visualisation)
        dt = 30 # ms
        timer_id = self.iren.CreateRepeatingTimer(dt)

    def addActor(self, actorWrapper):
        for actor in actorWrapper.actors:
            self.ren.AddActor(actor)


    def update_visualisation(self, obj=None, event=None):
        time.sleep(0.01)
        self.threadLock.acquire()
        self.ren.GetRenderWindow().Render()
        self.threadLock.release()


threadLock = threading.Lock()
viz = None

def show_cloud(points, color=[1,1,1], opacity=0.25):
    global viz
    if viz == None:
        viz = VTKVisualisation(threadLock)
    actorWrapper = VTKActorWrapper(points, color, opacity)
    #actorWrapper.update(threadLock, update_on)
    # actorWrapper.addPlane(W[i])
    viz.addActor(actorWrapper)


def show_planes(points, W):

    update_on = threading.Event()
    update_on.set()
    global viz
    if viz == None:
        viz = VTKVisualisation(threadLock)


    #pp = np.load('Pin.npy', allow_pickle=True)
    #W = np.load('W.npy')
    #pc = pp[1] - pp[1].mean(axis=0)


    #for i in range(len(points)):
    for i in range(10):
        actorWrapper = VTKActorWrapper(points[i], np.fabs(W[i, 0:3]))
        actorWrapper.update(threadLock, update_on)
        #actorWrapper.addPlane(W[i])
        viz.addActor(actorWrapper)

    #actorWrapper2 = VTKActorWrapper(points[1], [0, 1,0])
    #actorWrapper2.update(threadLock, update_on)




    #viz.addActor(actorWrapper2)

    viz.iren.Start()
    update_on.clear()


