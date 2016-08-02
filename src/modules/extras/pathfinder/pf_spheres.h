typedef struct sphere 
{ 
    const double vertex[3]; 
    const int* neighbours;
} sphere;

const sphere* get_sphere(int n);
