��OpenGL����ά�������������window�ϻ�ͼ
Software: Microsoft visual studio 2012 MFC

Library: opengl

���ȣ��Զ������

//Transform from screen coordination to OpenGL coordination
GLint viewport[4]; 
GLdouble modelview[16]; 
GLdouble projection[16]; 
GLfloat winX, winY, winZ; 
GLdouble posX, posY, posZ;

Ȼ����ת������

glGetIntegerv(GL_VIEWPORT, viewport);
glGetDoublev(GL_MODELVIEW_MATRIX, modelview); 
glGetDoublev(GL_PROJECTION_MATRIX, projection);

���õ���Ӧ��OpenGL����

 winX = (float)x; 
 winY = viewport[3] - (float)y;
 glReadPixels((int)winX, (int)winY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ); 
 gluUnProject(winX, winY, winZ, modelview, projection, viewport, &posX, &posY, &posZ); 

ע��(x, y)����Ļ���꣬(winX, winY, winZ)���Ӿ������꼰������꣬(posX, posY, posZ)��OpenGL���ꡣ

�˷�������glViewport(0, 0, screenWidth, screenHeight)����£�screenWidth��screenHeight�ֱ��ǿͻ����Ŀ�͸ߣ��ӿ����½�����ǡ���ǣ�0��0��������δ�����κ�ģ�ͱ任��

��������֤��

���Ǵ���������գ�http://blog.csdn.net/abcdef8c/article/details/6716737