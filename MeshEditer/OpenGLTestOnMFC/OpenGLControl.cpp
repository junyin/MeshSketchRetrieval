#include "stdafx.h"
#include "OpenGLControl.h"
#include ".\openglcontrol.h"
#include "MeshOperation.h"

#include <math.h>
#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <ANN/ANN.h>

using namespace std;					 // make std:: accessible

vector<MyMesh>  meshQueue;

bool NOISE_CONTROL = false;
bool NORMALIZE_CONTROL = false;
bool HISTOGRAM_CONTROL = false;
bool REMOVE_CONTROL = false;
bool PROJECTION_CONTROL = false;
bool SKETCH_CONTROL = false;
bool RETRIEVAL_CONTROL = false;
double noise_standard_deviation = 0.01;  //standard_deviation for adding noise
double mesh_histogram[1733] = {};
int PLOT_CONTROL = 1;
vector<double> sketchpoint_x;
vector<double> sketchpoint_y;
vector<double> sketchpoint_z;

COpenGLControl::COpenGLControl(void)
{
	m_fPosX = 0.0f;						 // X position of model in camera view
	m_fPosY = -0.1f;					 // Y position of model in camera view
	m_fZoom = 1.0f;						 // Zoom on model in camera view
	m_fRotX = 0.0f;						 // Rotation on model in camera view
	m_fRotY	= 0.0f;						 // Rotation on model in camera view
	m_bIsMaximized = false;
}

COpenGLControl::~COpenGLControl(void)
{
}

BEGIN_MESSAGE_MAP(COpenGLControl, CWnd)
	ON_WM_PAINT()
	ON_WM_SIZE()
	ON_WM_CREATE()
	ON_WM_TIMER()
	ON_WM_MOUSEMOVE()
END_MESSAGE_MAP()

void COpenGLControl::OnPaint()
{

	//CPaintDC dc(this); // device context for painting
	ValidateRect(NULL);
}

void COpenGLControl::OnSize(UINT nType, int cx, int cy)
{
	CWnd::OnSize(nType, cx, cy);

	if (0 >= cx || 0 >= cy || nType == SIZE_MINIMIZED) return;

	// Map the OpenGL coordinates.
	glViewport(0, 0, cx, cy);

	// Projection view
	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	// Set our current view perspective
	gluPerspective(35.0f, (float)cx / (float)cy, 0.01f, 2000.0f);

	// Model view
	glMatrixMode(GL_MODELVIEW);

	switch (nType)
	{
		// If window resize token is "maximize"
	case SIZE_MAXIMIZED:
		{
			// Get the current window rect
			GetWindowRect(m_rect);

			// Move the window accordingly
			MoveWindow(6, 6, cx - 14, cy - 14);

			// Get the new window rect
			GetWindowRect(m_rect);

			// Store our old window as the new rect
			m_oldWindow = m_rect;

			break;
		}

		// If window resize token is "restore"
	case SIZE_RESTORED:
		{
			// If the window is currently maximized
			if (m_bIsMaximized)
			{
				// Get the current window rect
				GetWindowRect(m_rect);

				// Move the window accordingly (to our stored old window)
				MoveWindow(m_oldWindow.left, m_oldWindow.top - 18, m_originalRect.Width() - 4, m_originalRect.Height() - 4);

				// Get the new window rect
				GetWindowRect(m_rect);

				// Store our old window as the new rect
				m_oldWindow = m_rect;
			}
			break;
		}
	}
}

int COpenGLControl::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CWnd::OnCreate(lpCreateStruct) == -1) return -1;

	oglInitialize();

	return 0;
}

void COpenGLControl::OnDraw(CDC *pDC)
{
	// If the current view is perspective...
	glLoadIdentity();
	gluLookAt(0.0,0.0,5.0,0.0,0.0,0.0,0.0,1.0,0.0);
	//glFrustum(-1, 1, -1, 1, 0.0, 40.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glTranslatef(0.0f, 0.0f, -m_fZoom);
	glTranslatef(m_fPosX, m_fPosY, 0.0f);
	glRotatef(m_fRotX, 1.0f, 0.0f, 0.0f);
	glRotatef(m_fRotY, 0.0f, 1.0f, 0.0f);
}

void COpenGLControl::OnTimer(UINT nIDEvent)
{
	switch (nIDEvent)
	{
	case 1:
		{
			// Clear color and depth buffer bits
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			// Draw OpenGL scene
			oglDrawScene();

			// Swap buffers
			SwapBuffers(hdc);

			break;
		}

	default:
		break;
	}

	CWnd::OnTimer(nIDEvent);
}

void COpenGLControl::OnMouseMove(UINT nFlags, CPoint point)
{
	int diffX = (int)(point.x - m_fLastX);
	int diffY = (int)(point.y - m_fLastY);
	m_fLastX  = (float)point.x;
	m_fLastY  = (float)point.y;

	// Left mouse button
	if (nFlags & MK_LBUTTON)
	{
		if(SKETCH_CONTROL)
		{

			
			//double pt_z = 0.5;
			//double camx = 0.1;
			//double camy = 0.1;
			//double camz = 0.5;
			//camx = camx*cos(m_fRotY*2*M_PI/360) -camz*sin(m_fRotY*2*M_PI/360);
			//camy = camx*sin(m_fRotX*2*M_PI/360)*sin(m_fRotY*2*M_PI/360)+camy*cos(m_fRotX*2*M_PI/360)+camz*sin(m_fRotX*2*M_PI/360)*cos(m_fRotY*2*M_PI/360);
			//camz = camx*cos(m_fRotX*2*M_PI/360)*sin(m_fRotY*2*M_PI/360)-camy*sin(m_fRotX*2*M_PI/360)+camz*cos(m_fRotX*2*M_PI/360)*cos(m_fRotY*2*M_PI/360);

			//pt_x = camx + pt_x;
			//pt_y = camy + pt_y;
			//pt_z = camz + pt_z;
			sketchpoint_x.push_back((double)point.x);
			sketchpoint_y.push_back((double)point.y);
			//sketchpoint_z.push_back(pt_z);

			//for(int i = 0 ;i<sketchpoint_x.size();i++){

			//	sketchpoint_x.at(i) = sketchpoint_x.at(i)*cos(m_fRotY*2*M_PI/360) -sketchpoint_z*sin(m_fRotY*2*M_PI/360) + m_fPosX;
			//	sketchpoint_y.at(i) =sketchpoint_x.at(i)*sin(m_fRotX*2*M_PI/360)*sin(m_fRotY*2*M_PI/360)+sketchpoint_y.at(i)*cos(m_fRotX*2*M_PI/360)+sketchpoint_z*sin(m_fRotX*2*M_PI/360)*cos(m_fRotY*2*M_PI/360)+m_fPosY;
			//	sketchpoint_z = sketchpoint_x.at(i)*cos(m_fRotX*2*M_PI/360)*sin(m_fRotY*2*M_PI/360)-sketchpoint_y.at(i)*sin(m_fRotY*2*M_PI/360)+sketchpoint_z*cos(m_fRotX*2*M_PI/360)*cos(m_fRotY*2*M_PI/360);
			//	//glVertex3f((float)sketchpoint.at(i).x,(float)sketchpoint.at(i).y,(float)0.0);
			//	//}
			//}

		}
		else{

			m_fRotX += (float)0.5f * diffY;

			if ((m_fRotX > 360.0f) || (m_fRotX < -360.0f))
			{
				m_fRotX = 0.0f;
			}

			m_fRotY += (float)0.5f * diffX;

			if ((m_fRotY > 360.0f) || (m_fRotY < -360.0f))
			{
				m_fRotY = 0.0f;
			}


		}
	}

	//if (nFlags & MK_LBUTTON && SKETCH_CONTROL%2==0)
	//{
	//	sketchpoint.push_back(point.x);
	//	//sketchpoint_y.push_back(point.y);
	//}

	// Middle mouse button
	else if (nFlags & MK_MBUTTON)
	{
		m_fZoom -= (float)0.01f * diffY;
	}

	// Right mouse button
	else if (nFlags & MK_RBUTTON)
	{
		m_fPosX += (float)0.0005f * diffX;
		m_fPosY -= (float)0.0005f * diffY;
	}

	OnDraw(NULL);

	CWnd::OnMouseMove(nFlags, point);
}

void COpenGLControl::oglCreate(CRect rect, CWnd *parent)
{
	CString className = AfxRegisterWndClass(CS_HREDRAW | CS_VREDRAW | CS_OWNDC, NULL, (HBRUSH)GetStockObject(BLACK_BRUSH), NULL);

	CreateEx(0, className, "OpenGL", WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, rect, parent, 0);

	// Set initial variables' values
	m_oldWindow	   = rect;
	m_originalRect = rect;

	hWnd = parent;
}

void COpenGLControl::oglInitialize(void)
{
	// Initial Setup:
	//
	static PIXELFORMATDESCRIPTOR pfd =
	{
		sizeof(PIXELFORMATDESCRIPTOR),
		1,
		PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
		PFD_TYPE_RGBA,
		32, // bit depth
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		16, // z-buffer depth
		0, 0, 0, 0, 0, 0, 0,
	};

	// Get device context only once.
	hdc = GetDC()->m_hDC;

	// Pixel format.
	m_nPixelFormat = ChoosePixelFormat(hdc, &pfd);
	SetPixelFormat(hdc, m_nPixelFormat, &pfd);

	// Create the OpenGL Rendering Context.
	hrc = wglCreateContext(hdc);
	wglMakeCurrent(hdc, hrc);

	// Basic Setup:
	//
	// Set color to use when clearing the background.
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearDepth(1.0f);
	////////////////////////////////////////

	///////////////////////////////////////
	// Turn on backface culling
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);

	// Turn on depth testing
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	// Send draw request
	OnDraw(NULL);


}

void COpenGLControl::oglDrawScene(void)
{

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	unsigned int meshsize = meshQueue.size();

	//add noise to current mesh
	if(NOISE_CONTROL && meshsize>=1)
	{
		AddNoise(noise_standard_deviation,meshQueue.at(meshsize-1));
	}
	if(REMOVE_CONTROL && meshsize>=1)
	{
		RemoveSameVertices(meshQueue.at(meshsize-1));
		/*MyMesh new_mesh;
		RemoveSameVertices2(meshQueue.at(meshsize-1),new_mesh);
		meshQueue.pop_back();
		meshQueue.push_back(new_mesh);*/
	}
	if(SKETCH_CONTROL)
	{
		//MeshRetrieval(MyMesh &mesh, double *sketchpoint_x,double *sketchpoint_y)
	}
	if(RETRIEVAL_CONTROL)
	{
		MeshRetrieval(meshQueue.at(meshsize-1),sketchpoint_x,sketchpoint_y);
	}
	//if(PROJECTION_CONTROL  && meshsize>=1)
	//{
	//	//float m_fRotX,float m_fRotY
	//	MyMesh projection_mesh=meshQueue.at(meshsize-1);
	//	MeshProjection(projection_mesh);

	//}
	if(NORMALIZE_CONTROL && meshsize>=1)
	{
		Normalizer(meshQueue.at(meshsize-1));
	}
	if(HISTOGRAM_CONTROL && meshsize>=1 && PLOT_CONTROL % 2 == 0)
	{
		GenMeshHistogram(meshQueue.at(meshsize-1),mesh_histogram);
	}

	//draw meshes
	for (unsigned int i=0;i<meshsize;i++)
	{
		if(meshsize>0)
		{
			if(SKETCH_CONTROL)
			{
				//sketch
				/*glBegin(GL_POLYGON);
				glVertex3f(0.0,0.0,0.0);
				glVertex3f(1.0,0.0,0.0);
				glVertex3f(1.0,1.0,0.0);
				glVertex3f(0.0,1.0,0.0);
				glEnd();*/
				m_fPosX = 0.0f;						 // X position of model in camera view
				m_fPosY = -0.1f;					 // Y position of model in camera view
				//m_fZoom = 1.0f;						 // Zoom on model in camera view
				m_fRotX = 0.0f;						 // Rotation on model in camera view
				m_fRotY	= 0.0f;						 // Rotation on model in camera view

				glColor3f(GLfloat(1.0), GLfloat(1.0), GLfloat(1.0));
				glBegin(GL_LINES);
				for(auto it = meshQueue.at(i).vertices_begin(); it != meshQueue.at(i).vertices_end(); ++it)
				{
					int index = it->idx();
					auto point = meshQueue.at(i).point(it.handle());
					glVertex3f((float)point.data()[0],(float)point.data()[1],0.0f);
				}
				glEnd();
				glColor3f(GLfloat(1.0), GLfloat(1.0), GLfloat(0.0));
				glPointSize(2.0);
				glBegin(GL_POINTS);
				for(int i = 0;i<sketchpoint_x.size();i++)
				{

					glVertex3f((float)(sketchpoint_x.at(i)/500.0-0.5),(float)(0.5-sketchpoint_y.at(i)/300.0),0.0f);
				}


				glEnd();
			}
			else
			{
				glColor3f(GLfloat(1.0), GLfloat(0.0), GLfloat(0.0));
				glBegin(GL_LINES);
				glVertex3f(0.0,0.0,0.0);
				glVertex3f(1.0,0.0,0.0);
				glEnd();
				glColor3f(GLfloat(0.0), GLfloat(1.0), GLfloat(0.0));
				glBegin(GL_LINES);
				glVertex3f(0.0,0.0,0.0);
				glVertex3f(0.0,1.0,0.0);
				glEnd();
				glColor3f(GLfloat(0.0), GLfloat(0.0), GLfloat(1.0));
				glBegin(GL_LINES);
				glVertex3f(0.0,0.0,0.0);
				glVertex3f(0.0,0.0,1.0);
				glEnd();

				glBegin(GL_POINTS);
				//change the colour for each mesh
				switch (i) 
				{
				case 0:
					glColor3f(GLfloat(1.0), GLfloat(0.8), GLfloat(0.6));
					break;
				case 1:
					glColor3f(GLfloat(0.7), GLfloat(0.5), GLfloat(1.0));
					break;
				case 2:
					glColor3f(GLfloat(0.6), GLfloat(1.0), GLfloat(0.5));
					break;
				case 3:
					glColor3f(GLfloat(0.6), GLfloat(1.0), GLfloat(1.0));
					break;
				default:
					glColor3f(GLfloat(0.5), GLfloat(0.5), GLfloat(0.5));
				};
				for (auto it = meshQueue.at(i).vertices_begin(); it != meshQueue.at(i).vertices_end(); ++it)
				{
					int index = it->idx();
					auto point = meshQueue.at(i).point(it.handle());
					glVertex3f(point.data()[0],point.data()[1],point.data()[2]);
				}
				glEnd();

				if(mesh_histogram[0]>0 && PLOT_CONTROL % 2 == 0)
				{
					glBegin(GL_LINES);
					glVertex3f(0.0,0.0,0.0);
					glVertex3f(1.0,0.0,0.0);
					glEnd();
					glBegin(GL_LINES);
					glVertex3f(0.0,0.0,0.0);
					glVertex3f(0.0,1.0,0.0);
					glEnd();
					//histogram
					glBegin(GL_LINES);
					for(int i = 0 ;i<1733;i++){
						glVertex3f(float(i)/1733.0,mesh_histogram[i],0.0);
					}
					glEnd();
				}
			}//end else
		}
	}//end for (unsigned int i=0;i<meshsize;i++)

}