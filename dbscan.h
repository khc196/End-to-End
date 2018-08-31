#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>
#include <stdio.h>
#define UNCLASSIFIED -1
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -2
#define SUCCESS 0
#define FAILURE -3

using namespace std;

struct cPoint
{
    int x, y, z;  // X, Y, Z position
    int clusterID;  // clustered ID
    cPoint(int _x, int _y, int _z){
        x = _x;
        y = _y;
        z = _z;
        clusterID = UNCLASSIFIED;
    }
};

class DBSCAN {
public:    
    DBSCAN(unsigned int minPts, float eps, vector<cPoint> points){
        m_minPoints = minPts;
        m_epsilon = eps;
        m_points = points;
        m_pointSize = points.size();
    }
    ~DBSCAN(){}

    int run();
    vector<int> calculateCluster(cPoint point);
    int expandCluster(cPoint point, int clusterID);
    inline double calculateDistance(cPoint pointCore, cPoint pointTarget);

    int getTotalPointSize() {return m_pointSize;}
    int getMinimumClusterSize() {return m_minPoints;}
    int getEpsilonSize() {return m_epsilon;}
    vector<cPoint> m_points;
private:
    unsigned int m_pointSize;
    unsigned int m_minPoints;
    float m_epsilon;
};

#endif // DBSCAN_H