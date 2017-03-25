//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================


#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
    int logLen;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0) {
        char * log = new char[logLen];
        int written;
        glGetShaderInfoLog(handle, logLen, &written, log);
        printf("Shader log:\n%s", log);
        delete log;
    }
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK) {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}

// check if shader could be linked
void checkLinking(unsigned int program) {
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK) {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec3 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

class vec3 {

public:
    float x, y, z;

    vec3() : x(0), y(0), z(0) {}

    vec3(float x, float y) {
        this->x = x;
        this->y = y;
        this->z = 0;
    }

    vec3(float x, float y, float z) {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    vec3(vec3 const & vec) {
        x = vec.x;
        y = vec.y;
        z = vec.z;
    }

    vec3 operator*(float n) {
        return vec3(
                x * n,
                y * n,
                z * n
        );
    }

    vec3 operator/(float n) {
        return n == 0.0f
               ? vec3(x, y, z)
               : vec3(
                x / n,
                y / n,
                z / n
        );
    }

    vec3 operator*(double n) {
        return vec3(
                x * (float) n,
                y * (float) n,
                z * (float) n
        );
    }

    vec3 operator+(vec3 const & vec) {
        return vec3(
                x + vec.x,
                y + vec.y,
                z + vec.z
        );
    }

    vec3 operator*(vec3 const & vec) {
        return vec3(
                x * vec.x,
                y * vec.y,
                z * vec.z
        );
    }

    void operator+=(vec3 const & vec) {
        this->x += vec.x;
        this->y += vec.y;
        this->z += vec.z;
    }

    vec3 cross(vec3 const & vec) {
        return vec3(
            y * vec.z - z * vec.y,
            z * vec.x - x * vec.z,
            x * vec.y - y * vec.x
        );
    }
};

// row-major matrix 4x4
struct mat4 {
    float m[4][4];
public:
    mat4() {}
    mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }

    mat4 operator*(const mat4& right) {
        mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
            }
        }
        return result;
    }
    operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
    float v[4];

    vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
        v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }

    vec4 operator*(const mat4& mat) {
        vec4 result;
        for (int j = 0; j < 4; j++) {
            result.v[j] = 0;
            for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
        }
        return result;
    }
};

// 2D camera
struct Camera {
    float wCx, wCy;	// center in world coordinates
    float wWx, wWy;	// width and height in world coordinates
public:
    Camera() {
        Animate(0);
    }

    mat4 V() { // view matrix: translates the center to the origin
        return mat4(1,    0, 0, 0,
                    0,    1, 0, 0,
                    0,    0, 1, 0,
                    -wCx, -wCy, 0, 1);
    }

    mat4 P() { // projection matrix: scales it to be a square of edge length 2
        return mat4(2/wWx,    0, 0, 0,
                    0,    2/wWy, 0, 0,
                    0,        0, 1, 0,
                    0,        0, 0, 1);
    }

    mat4 Vinv() { // inverse view matrix
        return mat4(1,     0, 0, 0,
                    0,     1, 0, 0,
                    0,     0, 1, 0,
                    wCx, wCy, 0, 1);
    }

    mat4 Pinv() { // inverse projection matrix
        return mat4(wWx/2, 0,    0, 0,
                    0, wWy/2, 0, 0,
                    0,  0,    1, 0,
                    0,  0,    0, 1);
    }

    void Animate(float t) {
        wCx = 0; // 10 * cosf(t);
        wCy = 0;
        wWx = 20;
        wWy = 20;
    }
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

class Triangle {
    unsigned int vao;	// vertex array object id
    float sx, sy;		// scaling
    float wTx, wTy;		// translation
public:
    Triangle() {
        Animate(0);
    }

    void Create() {
        glGenVertexArrays(1, &vao);	// create 1 vertex array object
        glBindVertexArray(vao);		// make it active

        unsigned int vbo[2];		// vertex buffer objects
        glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

        // vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
        static float vertexCoords[] = {
                -10, -10, // left top
                -10, 10, // left bottom
                10, -10 // right bottom
        };	// vertex data on the CPU

        glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
                     sizeof(vertexCoords), // number of the vbo in bytes
                     vertexCoords,		   // address of the data array on the CPU
                     GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
        // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
        glEnableVertexAttribArray(0);
        // Data organization of Attribute Array 0 
        glVertexAttribPointer(0,			// Attribute Array 0
                              2, GL_FLOAT,  // components/attribute, component type
                              GL_FALSE,		// not in fixed point format, do not normalized
                              0, NULL);     // stride and offset: it is tightly packed

        // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
        static float vertexColors[] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };	// vertex data on the CPU
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

        // Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
        glEnableVertexAttribArray(1);  // Vertex position
        // Data organization of Attribute Array 1
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
    }

    void Animate(float t) {
        sx = 1; // sinf(t);
        sy = 1; // cosf(t);
        wTx = 0; // 4 * cosf(t / 2);
        wTy = 0; // 4 * sinf(t / 2);
    }

    void Draw() {
        mat4 Mscale(sx, 0, 0, 0,
                    0, sy, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 1); // model matrix

        mat4 Mtranslate(1,   0,  0, 0,
                        0,   1,  0, 0,
                        0,   0,  0, 0,
                        wTx, wTy,  0, 1); // model matrix

        //   Mscale * Mtranslate *
        mat4 MVPTransform = camera.V() * camera.P();

        // set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
        int location = glGetUniformLocation(shaderProgram, "MVP");
        if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
        else printf("uniform MVP cannot be set\n");

        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
    }
};

class LineStrip {
    GLuint vao;        // vertex array object, vertex buffer object // vbo
    float  vertexData[100]; // interleaved data of coordinates and colors
    int    nVertices;       // number of vertices
public:
    LineStrip() {
        nVertices = 0;
    }
    void Create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        GLuint vbo;	// vertex/index buffer object
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0
        glEnableVertexAttribArray(1);  // attribute array 1
        // Map attribute array 0 to the vertex data of the interleaved vbo
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
        // Map attribute array 1 to the color data of the interleaved vbo
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
    }

    void AddPoint(float cX, float cY) {
        if (nVertices >= 20) return;

        vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
        // fill interleaved data
        vertexData[5 * nVertices]     = cX; //wVertex.v[0];
        vertexData[5 * nVertices + 1] = cY; //wVertex.v[1];
        vertexData[5 * nVertices + 2] = 0; // red
        vertexData[5 * nVertices + 3] = 1; // green
        vertexData[5 * nVertices + 4] = 0; // blue
        nVertices++;
        // copy data to the GPU
        glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
    }

    void Draw() {
        if (nVertices > 0) {
            mat4 VPTransform = camera.V() * camera.P();

            int location = glGetUniformLocation(shaderProgram, "MVP");
            if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
            else printf("uniform MVP cannot be set\n");

            glBindVertexArray(vao);
            glDrawArrays(GL_LINE_STRIP, 0, nVertices);
        }
    }
};

class BezierCurve {
    static const int sMaxVertices = 100;

    GLuint vao;
    float vertexData[sMaxVertices];
    int nVertices;

    std::vector<vec3> cps;	// control points

    float B(int i, float t) {
        size_t n = cps.size() - 1; // n deg polynomial = n+1 pts!

        float choose = 1;

        for(int j = 1; j <= i; j++) {
            choose *= (float)(n - j + 1) / j;
        }

        return (float)(choose * pow(t, i) * pow(1 - t, n - i));
    }

public:
    BezierCurve() : nVertices(0) {}

    void Create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        GLuint vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
    }

    void AddControlPoint(vec3 cp) {
        cps.push_back(cp);
    }

    vec3 r(float t) {
        vec3 rr(0, 0);

        for(int i = 0; i < cps.size(); i++) {
            rr += cps[i] * B(i, t);
        }

        return rr;
    }

    void Generate() {

        for(float t = 0; t < 1.0f; t += (1.0f / 20)) {
            vec3 vec = r(t);

            //vec4 wVertex = vec4(vec.x, vec.y, 0, 1) * camera.Pinv() * camera.Vinv();

            //printf("vec: %f %f\n", vec.x, vec.y);

            vertexData[5 * nVertices]     = vec.x; //wVertex.v[0];
            vertexData[5 * nVertices + 1] = vec.y; //wVertex.v[1];
            vertexData[5 * nVertices + 2] = 1; // red
            vertexData[5 * nVertices + 3] = 1; // green
            vertexData[5 * nVertices + 4] = 0; // blue
            nVertices++;
        }

        printf("%p", vertexData);
        glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
    }

    void Draw() {
        mat4 VPTransform = camera.V() * camera.P();

        int location = glGetUniformLocation(shaderProgram, "MVP");

        if (location >= 0)
            glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
        else
            printf("uniform MVP cannot be set\n");

        glBindVertexArray(vao);
        glDrawArrays(GL_LINE_STRIP, 0, nVertices);
    }
};

class BezierSurface {
    GLuint vao;
    GLuint vbo[2];

    unsigned int nVertices;
    float vertices[16 * 20 * 5 * sizeof(float)];

    std::vector<std::vector<vec3>> controlPoints;
    std::vector<std::vector<vec3>> points;

    struct VertexData {
        vec3 position, normal;
        float u, v;
    };

    size_t factorial(unsigned int n) {
        size_t result = 1;

        for(unsigned int i = 1; i <= n; i++) {
            result *= i;
        }

        return result;
    }

    size_t binomial(unsigned int n, unsigned int i) {
        return factorial(n) / (factorial(i) * factorial(n - i));
    }

    double Bernstein(unsigned int n, unsigned int i, float u) {
        return binomial(n, i) * pow(u, i) * pow(1 - u, n - i);
    }

    VertexData p(float u, float v) {
        vec3 vec;

        unsigned int n = (unsigned int)controlPoints.size();
        unsigned int m = n;


        for(unsigned int i = 0; i < n; i++) {
            for(unsigned int j = 0; j < m; j++) {
                //printf("%d/%d %d/%d %f %f\n", i, n, j, m, u, v);
                vec += (controlPoints[i][j] * Bernstein(n, i, u) * Bernstein(m, j, v));
            }
        }

        /*
        for(unsigned int i = 0; i < n; i++) {
            vec3 tmp;

            for(unsigned int j = 0; j < m; j++) {
                //printf("%d/%d %d/%d %f %f\n", i, n, j, m, u, v);
                tmp += (controlPoints[i][j] * Bernstein(m, j, v));
            }

            vec += tmp * Bernstein(n, i, u);
        }
*/
        /*
        vec3 v1, v2;

        for(unsigned int i = 0; i < n; i++) {
            v1 += controlPoints[0][i] * Bernstein(n, i, u);
        }

        for(unsigned int i = 0; i < n; i++) {
            v2 += controlPoints[i][0] * Bernstein(n, i, v);
        }

        vec = v1 * v2;
         */


        vec4 wVertex = vec4(vec.x, vec.y, vec.z, 1) * camera.Pinv() * camera.Vinv();
        vec3 pos = vec; //vec3(wVertex.v[0], wVertex.v[1], 0);


        vec3 a = vec3(vec / u);
        vec3 b = vec3(vec / v);

        vec3 cross = a.cross(b);

        /*
        printf("pos: %f %f %f\n", pos.x, pos.y, pos.z);
        printf("a__: %f %f %f\n", a.x, a.y, a.z);
        printf("b__: %f %f %f\n", b.x, b.y, b.z);
        printf("cro: %f %f %f\n", cross.x, cross.y, cross.z);
*/
        return {pos, cross, u, v};
    }

public:
    BezierSurface() : nVertices(0) {};

    void Create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(2, &vbo[0]);
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

        /*
        unsigned int N = controlPoints.size();
        nVertices = N * N;

        VertexData * vtx = new VertexData[nVertices], *pVtx = vtx;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                *pVtx++ = {controlPoints[i][j], {1, 0, 0}, 1.0f, 1.0f};
            }
        }
*/

        unsigned int N = 6; //controlPoints.size();
        nVertices = N * N * 6;

        VertexData * vtx = new VertexData[nVertices], *pVtx = vtx;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                *pVtx++ = p((float)i / N,        (float)j / N);
                *pVtx++ = p((float)(i + 1) / N,  (float)j / N);
                *pVtx++ = p((float)i / N,        (float)(j + 1) / N);
                *pVtx++ = p((float)(i + 1) / N,  (float)j / N);
                *pVtx++ = p((float)(i + 1) / N,  (float)(j + 1) / N);
                *pVtx++ = p((float)i / N,        (float)(j + 1) / N);
            }
        }

        glBufferData(GL_ARRAY_BUFFER, nVertices * sizeof(VertexData), vtx, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);  // AttribArray 0 = POSITION
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)0);

        glEnableVertexAttribArray(1);  // AttribArray 1 = NORMAL
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));

        glEnableVertexAttribArray(2);  // AttribArray 2 = UV
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, u));

        float vertexColors[3 * nVertices];

        for(auto i = 0; i < nVertices; i++) {
            vertexColors[i * 3] = 1.0f / (i + 1);
            vertexColors[i * 3 + 1] = 0 + (1.0f / (i+1));
            vertexColors[i * 3 + 2] = 1.0f - (1.0f / (i+1));
        }

        printf("%d %d\n", sizeof(vertexColors), nVertices * 3 * sizeof(float));
        printf("%d %d %d %d %d\n", factorial(0), factorial(1), factorial(2), factorial(3), factorial(4));

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void Generate() {
        // 16 control points
        auto n = 4;
        float delta = 5.0f; //10.0f / n;

        for(int i = 0; i < n; i++) {

            std::vector<vec3> tmp;
            for(int j = 0; j < n; j++) {
                tmp.push_back(vec3(-10.0f + i * delta, -10.0f + j * delta, 0));
            }

            controlPoints.push_back(tmp);
        }

        for(auto i = 0; i < n; i++) {
            for(auto j = 0; j < n; j++) {
                printf("%d %d: %f %f %f\n", i, j, controlPoints[i][j].x, controlPoints[i][j].y, controlPoints[i][j].z);
            }
        }
    }

    void Draw() {
        mat4 VPTransform = camera.V() * camera.P();

        int location = glGetUniformLocation(shaderProgram, "MVP");

        if (location >= 0)
            glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
        else
            printf("uniform MVP cannot be set\n");

        glBindVertexArray(vao);
        glDrawArrays(GL_LINES, 0, nVertices); // GL_LINE_STRIP GL_TRIANGLES
    }
};

class LagrangeCurve {
    static const int sMaxVertices = 100;

    GLuint vao;

    float vertexData[sMaxVertices * 5];
    int nVertices;

    std::vector<vec3>  cps;	// control points
    std::vector<float> ts; 	// parameter (knot) values

    float L(int i, float t) {
        float Li = 1.0f;

        for(int j = 0; j < cps.size(); j++) {
            if (j != i)
                Li *= (t - ts[j]) / (ts[i] - ts[j]);
        }

        return Li;
    }

public:

    LagrangeCurve() : nVertices(0) {};

    void Create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
    }

    // https://www.it.uu.se/edu/course/homepage/grafik1/ht07/examples/curves.cpp
    void AddControlPoint(vec3 cp) {
        float ti = cps.size(); 	// or something better

        cps.push_back(cp);
        ts.push_back(ti);
    }

    vec3 r(float t) {
        vec3 rr(0, 0, 0);

        for(int i = 0; i < cps.size(); i++) {
            rr += cps[i] * L(i, t);
        }

        return rr;
    }

    void Generate() {
        if( cps.size() < 3 )
            return;


        nVertices = 0;

        for(float t = 0.0f; t < cps.size() - 1; t += 0.05f) {
            vec3 vec = r(t);

            vec4 wVertex = vec4(vec.x, vec.y, 0, 1) * camera.Pinv() * camera.Vinv();

            vertexData[5 * nVertices]     = wVertex.v[0];
            vertexData[5 * nVertices + 1] = wVertex.v[1];
            vertexData[5 * nVertices + 2] = 1; // red
            vertexData[5 * nVertices + 3] = 1; // green
            vertexData[5 * nVertices + 4] = 0; // blue
            nVertices++;
        }

        glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
    }

    void Draw() {
        if( cps.size() < 3 )
            return;

        mat4 VPTransform = camera.V() * camera.P();

        int location = glGetUniformLocation(shaderProgram, "MVP");

        if (location >= 0)
            glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
        else
            printf("uniform MVP cannot be set\n");


        glBindVertexArray(vao);
        glDrawArrays(GL_LINE_STRIP, 0, nVertices);
    }

};

class Arrow {
    unsigned int vao;
    unsigned int vbo[2];

    vec3 position;

    bool display;

public:
    Arrow() : display(false) {}

    void Generate() {
        static const float vertexCoords[] = {
                0, 0, 0,
                0.7f, -1.5f, 0,
                0, -1, 0,
                0, 0, 0,
                -0.7f, -1.5f, 0,
                0, -1, 0
        };

        static const float vertexColors[] = {
                0.7f, 0, 0,
                0, 0.7f, 0,
                0, 0, 0.7f,
                0.7f, 0, 0,
                0, 0.7f, 0,
                0, 0, 0.7f,
        };

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(2, &vbo[0]);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void Draw() {
        if( !Arrow::display )
            return;

        mat4 Mtranslate(1,   0,  0, 0,
                        0,   1,  0, 0,
                        0,   0,  0, 0,
                        position.x, position.y,  0, 1);

        mat4 MVPTransform = Mtranslate * camera.V() * camera.P();

        int location = glGetUniformLocation(shaderProgram, "MVP");

        if (location >= 0)
            glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform);
        else
            printf("uniform MVP cannot be set\n");

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    void show() {
        display = true;
    }

    void hide() {
        display = false;
    }
};

// The virtual world: collection of two objects
Triangle triangle;
LineStrip lineStrip;
BezierCurve bc;
BezierCurve bc2;
LagrangeCurve lagrangeCurve;
BezierSurface bezierSurface;
Arrow arrow;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);

    // Create objects by setting up their vertex data on the GPU
    triangle.Create();
    //lineStrip.Create();
    //bc.Create();

   // lagrangeCurve.Create();

    //arrow.Generate();
    //arrow.show();

    bezierSurface.Generate();
    bezierSurface.Create();




    //-8, -8, -6, 10, 8, -2
    /*
    bc.AddControlPoint(vec3(-8, -8));
    bc.AddControlPoint(vec3(-6, 10));
    bc.AddControlPoint(vec3(8, -2));

    bc2.AddControlPoint(vec3(-10, -9));
    bc2.AddControlPoint(vec3(-5, 8));
    bc2.AddControlPoint(vec3(7, -3));

    lineStrip.AddPoint(-8, -8);
    lineStrip.AddPoint(-6, 10);
    lineStrip.AddPoint(8, -2);
*/
    //bc.Generate();
    //bc2.Generate();

    // Create vertex shader from string
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    if (!vertexShader) {
        printf("Error in vertex shader creation\n");
        exit(1);
    }

    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    checkShader(vertexShader, (char *) "Vertex shader error");

    // Create fragment shader from string
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (!fragmentShader) {
        printf("Error in fragment shader creation\n");
        exit(1);
    }
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader, (char *) "Fragment shader error");

    // Attach shaders to a single program
    shaderProgram = glCreateProgram();
    if (!shaderProgram) {
        printf("Error in shader program creation\n");
        exit(1);
    }
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    // Connect the fragmentColor to the frame buffer memory
    glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

    // program packaging
    glLinkProgram(shaderProgram);
    checkLinking(shaderProgram);
    // make this program run
    glUseProgram(shaderProgram);
}

void onExit() {
    glDeleteProgram(shaderProgram);
    printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);							// background color 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

   // triangle.Draw();
    //lineStrip.Draw();
    //bc.Draw();
    //bc2.Draw();
    //printf("draw\n");
    //lagrangeCurve.Draw();
    bezierSurface.Draw();
   // arrow.Draw();

    glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'd')
        glutPostRedisplay();         // if d, invalidate display, i.e. redraw
    else if( key == 'q' )
        exit(0);
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
        float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
        float cY = 1.0f - 2.0f * pY / windowHeight;
        //lineStrip.AddPoint(cX, cY);
        //printf("%d %d\n", pX, pY);
        lagrangeCurve.AddControlPoint(vec3(cX, cY));
        lagrangeCurve.Generate();

        glutPostRedisplay();     // redraw
    }
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
    float sec = time / 1000.0f;				// convert msec to sec
    camera.Animate(sec);					// animate the camera
    //triangle.Animate(sec);					// animate the triangle object
    glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
    glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
    glewExperimental = GL_TRUE;	// magic : true
    glewInit();
#endif

    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    onInitialization();

    glutDisplayFunc(onDisplay);                // Register event handlers
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);

    glutMainLoop();
    onExit();
    return 1;
}
