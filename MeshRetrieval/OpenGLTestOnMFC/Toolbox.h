#include <math.h>
#include <stdio.h>
#include <random>
#pragma once
#include "afxwin.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

#undef min
#undef max
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/System/config.h>
#include <OpenMesh/Core/Mesh/Status.hh>
#include <OpenMesh/Core/IO/exporter/ExporterT.hh>

using namespace std; // make std:: accessible
typedef OpenMesh::PolyMesh_ArrayKernelT<>  MyMesh;

void loadHistogram(string filname,double *histogram);
double similarity(double *histogram_test,double *histogram_sketch);
void qsort_getid(double array[],double id_array[], int left_id, int right_id);
void swap(double array[], int i, int j);
double round(double number);
double FindMaxDistance(MyMesh &mesh);