#include "stdafx.h"
#include "OpenGLControl.h"
#include ".\openglcontrol.h"
#include "MeshOperation.h"

#include <math.h>
#include <stdio.h>

// make std:: accessible
using namespace std;

//Queue for all meshes
vector<MyMesh>  meshQueue;

bool NOISE_CONTROL = false;
bool NORMALIZE_CONTROL = false;
bool SKETCH_CONTROL = false;
int RETRIEVAL_CONTROL = 0; //0=no;1=back;2=seat;

//add noise variable
double noise_standard_deviation = 0.01; 

//retrieval variables
vector<double> sketchpoint_x;
vector<double> sketchpoint_y;
vector<double> grid_id_x;
vector<double> grid_id_y;
vector<MyMesh> projectedQueue;

//shading parameters
GLfloat mat_specular[]={1.0f, 0.0f, 1.0f, 1.0f};
GLfloat mat_diffuse[]={0.8f, 0.8f, 0.8f, 1.0f};
GLfloat mat_ambient[]={0.8f, 0.8f, 0.8f, 1.0f};
GLfloat mat_shininess[]={80.0};

//transform from screen coordination to OpenGL coordination
GLint    viewport[4]; 
GLdouble modelview[16]; 
GLdouble projection[16]; 
GLfloat  winX, winY, winZ; 
GLdouble posX, posY, posZ;

double theta_x = 0.0;
double theta_y = 0.0;

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
	gluLookAt(0.0,0.0,2.0,0.0,0.0,0.0,0.0,1.0,0.0);
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

			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
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
			sketchpoint_x.push_back((double)point.x);
			sketchpoint_y.push_back((double)point.y);
		}
		else
		{
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

	GLfloat light_ambient1[]={0.0f, 0.0f, 0.0f, 1.0f};
	GLfloat light_diffuse1[]={1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular1[]={0.8f, 0.8f, 0.8f, 1.0f};
	GLfloat light_position1[]={11.0, 11.0, 11.0, 0.0};

	glLightfv(GL_LIGHT0, GL_POSITION, light_position1);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient1);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse1);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular1);
	glEnable(GL_LIGHT0);

	glMaterialfv(GL_FRONT,GL_AMBIENT,mat_ambient);
	glMaterialfv(GL_FRONT,GL_DIFFUSE,mat_diffuse);
	glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular);
	glMaterialfv(GL_FRONT,GL_SHININESS,mat_shininess);

	glEnable(GL_LIGHTING);

	glShadeModel(GL_FLAT);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
	///////////////////////////////////////
	// Turn on backface culling
	//glFrontFace(GL_CCW);
	//glCullFace(GL_BACK);

	// Turn on depth testing

	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0, 0.0, 2.0,  /* eye is at (0,0,1) */
		0.0, 0.0, 0.0,        /* center is at (0,0,0) */
		0.0, 1.0, 0.0);       /* up is in positive Y direction */

	//load initial mesh 
	string init_mesh_filname = "./MeshData/initial.obj";
	MyMesh init_mesh;
	OpenMesh::IO::read_mesh(init_mesh, init_mesh_filname);
	meshQueue.push_back(init_mesh);

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
	//normalize the current mesh
	if(NORMALIZE_CONTROL && meshsize>=1)
	{
		Normalizer(meshQueue.at(meshsize-1));
	}
	//retrieval
	if(RETRIEVAL_CONTROL!=0 && !sketchpoint_x.empty())
	{
		grid_id_x.clear();
		grid_id_y.clear();
		MeshSketchRetrieval(sketchpoint_x,sketchpoint_y,grid_id_x,grid_id_y);
		sketchpoint_x.clear();
		sketchpoint_y.clear();
	}

	//get 2D peojected mesh queue
	if(!SKETCH_CONTROL && meshsize>0)
	{
		projectedQueue.clear();
		for (unsigned int i = 0;i<meshsize;i++)
		{
			//Restore rotated 3D object
			projectedQueue.push_back(meshQueue.at(i));
		}
	}
	//rotate the mesh and get corresponding 2D projection
	if(SKETCH_CONTROL && meshsize>0)
	{
		theta_x = -m_fRotX*2*M_PI/360.0;
		theta_y = m_fRotY*2*M_PI/360.0;
		/*double theta_x = -m_fRotX*2*M_PI/360.0;
		double theta_y = m_fRotY*2*M_PI/360.0;*/
		for (int i=0;i<meshsize;i++)
		{
			for(MyMesh::VertexIter v_it = projectedQueue.at(i).vertices_begin();v_it!=projectedQueue.at(i).vertices_end();++v_it)
			{
				//rotate the mesh and show in 2D projection
				MyMesh::Point point = projectedQueue.at(i).point(v_it.handle());
				double temp_x = point.data()[0],temp_y = point.data()[1],temp_z = point.data()[2];
				(float)*(projectedQueue.at(i).point(v_it).data()+0) =  temp_x*cos(theta_y)+temp_y*sin(theta_x)*sin(theta_y)+temp_z*cos(theta_x)*sin(theta_y);
				(float)*(projectedQueue.at(i).point(v_it).data()+1) =  temp_y*cos(theta_x)+temp_z*sin(theta_x);
				(float)*(projectedQueue.at(i).point(v_it).data()+2) = -temp_x*sin(theta_y)+temp_y*sin(theta_x)*cos(theta_y)+temp_z*cos(theta_x)*cos(theta_y);
				glVertex3f((float)*(projectedQueue.at(i).point(v_it).data()+0),(float)*(projectedQueue.at(i).point(v_it).data()+1),0.0f);
			}
		}
	}

	/*Draw Meshes*/
	for (unsigned int i=0;i<meshsize;i++)
	{
		/*Show Sketch*/
		//Get 2D projected view of rotated 3D object
		if(SKETCH_CONTROL)
		{
			glDisable(GL_LIGHTING);
			//draw the projection of the current mesh at xy plane 
			glColor3f(GLfloat(0.6), GLfloat(0.6), GLfloat(0.6));
			glBegin(GL_POINTS);
			for(MyMesh::VertexIter v_it = projectedQueue.at(i).vertices_begin();v_it!=projectedQueue.at(i).vertices_end();++v_it)
			{
				//rotate the mesh and show in 2D projection
				glVertex3f((float)*(projectedQueue.at(i).point(v_it).data()+0),(float)*(projectedQueue.at(i).point(v_it).data()+1),0.0f);
			}
			glEnd();

			//draw the sketch
			glGetIntegerv(GL_VIEWPORT, viewport);
			glGetDoublev(GL_MODELVIEW_MATRIX, modelview); 
			glGetDoublev(GL_PROJECTION_MATRIX, projection);

			glColor3f(GLfloat(1.0), GLfloat(1.0), GLfloat(0.0));
			glPointSize(2.0);
			glBegin(GL_POINTS);
			for(unsigned int i = 0;i<sketchpoint_x.size();i++)
			{
				winX = (float)sketchpoint_x.at(i); 
				winY = viewport[3] -(float)sketchpoint_y.at(i);
				glReadPixels((int)winX, (int)winY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ); 
				gluUnProject(winX, winY, winZ, modelview, projection, viewport, &posX, &posY, &posZ);
				glVertex3f(posX,posY,posZ);
			}
			glEnd();

			//initial the factors
			m_fPosX =  0.0f;						 // X position of model in camera view
			m_fPosY = -0.1f;					     // Y position of model in camera view
			m_fZoom =  1.0f;						 // Zoom on model in camera view
			m_fRotX =  0.0f;						 // Rotation on model in camera view
			m_fRotY	=  0.0f;						 // Rotation on model in camera view
		}
		else
		{ 
			//x axis
			glColor3f(GLfloat(1.0), GLfloat(0.0), GLfloat(0.0));
			glBegin(GL_LINES);
			glVertex3f(0.0,0.0,0.0);
			glVertex3f(1.0,0.0,0.0);
			glEnd();
			//y axis
			glColor3f(GLfloat(0.0), GLfloat(1.0), GLfloat(0.0));
			glBegin(GL_LINES);
			glVertex3f(0.0,0.0,0.0);
			glVertex3f(0.0,1.0,0.0);
			glEnd();
			//z axis
			glColor3f(GLfloat(0.0), GLfloat(0.0), GLfloat(1.0));
			glBegin(GL_LINES);
			glVertex3f(0.0,0.0,0.0);
			glVertex3f(0.0,0.0,1.0);
			glEnd();

			glEnable(GL_LIGHTING);
			//change the colour for each mesh
			switch (i) 
			{
			case 0:
				glColor3f(GLfloat(1.0), GLfloat(1.0), GLfloat(1.0));
				break;
			case 1:
				glColor3f(GLfloat(0.6), GLfloat(0.8), GLfloat(1.0));
				break;
			case 2:
				glColor3f(GLfloat(1.0), GLfloat(1.0), GLfloat(1.0));
				break;
			case 3:
				glColor3f(GLfloat(0.6), GLfloat(1.0), GLfloat(1.0));
				break;
			default:
				glColor3f(GLfloat(0.5), GLfloat(0.5), GLfloat(0.5));
			};

			meshQueue.at(i).request_face_normals();
			meshQueue.at(i).update_normals();

			GLdouble norms[3]={0.0,0.0,0.0};
			for(MyMesh::FaceIter f_it=meshQueue.at(i).faces_begin();f_it!=meshQueue.at(i).faces_end();++f_it)
			{
				for(int n=0;n<3;n++){
					norms[n]=(GLdouble)*(meshQueue.at(i).normal(f_it).data()+n);
				}
				glNormal3dv(norms);

				glPushMatrix();
				glBegin(GL_POLYGON);
				for(MyMesh::FaceVertexIter v_it=meshQueue.at(i).fv_iter(f_it);v_it;++v_it)
				{
					glVertex3fv(meshQueue.at(i).point(v_it).data());
				}
				glEnd();
				glPopMatrix();
			}

			//draw grid
			if(grid_id_x.size()>0)
			{
				//grid
				glColor3f(GLfloat(0.0), GLfloat(1.0), GLfloat(1.0));
				glPointSize(2.0);
				glBegin(GL_POINTS);
				for(unsigned int i = 0 ;i<grid_id_x.size();i++){
					glVertex3f(float(grid_id_x.at(i))/31,float(grid_id_y.at(i))/31,0.0);
				}
				glEnd();
			}//end draw grid

			//release the face normals
			meshQueue.at(i).release_face_normals();

		}//end else
	}//end for (unsigned int i=0;i<meshsize;i++)
}//end void COpenGLControl::oglDrawScene(void)

