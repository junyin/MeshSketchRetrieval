#pragma once
#include "afxwin.h"

#include <gl/gl.h>
#include <gl/glu.h>
#include <gl/glut.h>
#include <vector>

#undef min
#undef max
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
using namespace std;

typedef OpenMesh::PolyMesh_ArrayKernelT<>  MyMesh;

extern  vector<MyMesh> meshQueue;
extern bool NOISE_CONTROL;
extern bool REMOVE_CONTROL;
extern bool NORMALIZE_CONTROL;
extern bool HISTOGRAM_CONTROL;
extern bool SKETCH_CONTROL;
extern int PLOT_CONTROL;
extern bool PROJECTION_CONTROL;
extern bool RETRIEVAL_CONTROL;

extern vector<double> sketchpoint_x;
extern vector<double> sketchpoint_y;
extern vector<double> sketchpoint_z;
//extern vector<float> sketchpoint_y;

class COpenGLControl : public CWnd
{
public:
	/******************/
	/* Public Members */
	/******************/
	UINT_PTR m_unpTimer;
	// View information variables
	float	 m_fLastX;
	float	 m_fLastY;
	float	 m_fPosX;
	float	 m_fPosY;
	float	 m_fZoom;
	float	 m_fRotX;
	float	 m_fRotY;
	bool	 m_bIsMaximized;

private:
	/*******************/
	/* Private Members */
	/*******************/
	// Window information
	CWnd  *hWnd;
	HDC   hdc;			
	HGLRC hrc;			
	int   m_nPixelFormat;
	CRect m_rect;
	CRect m_oldWindow;
	CRect m_originalRect;

public:
	COpenGLControl(void);
	virtual ~COpenGLControl(void);

	void oglCreate(CRect rect, CWnd *parent);
	void oglInitialize(void);
	void oglDrawScene(void);

	// Added message classes:
	afx_msg void OnPaint();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg	void OnDraw(CDC *pDC);
	afx_msg int  OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnTimer(UINT nIDEvent);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);

	DECLARE_MESSAGE_MAP()
};
