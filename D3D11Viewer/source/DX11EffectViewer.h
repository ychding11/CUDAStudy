#pragma once
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#if D3D_COMPILER_VERSION < 46
#include <d3dx11.h>
#endif

#include <cstdio>

// code migration. https://msdn.microsoft.com/en-us/library/windows/desktop/ee418730(v=vs.85).aspx
#include <DirectXMath.h>

using namespace DirectX;


#define CHECK_D3D11_CALL(x)                           \
do{                                                   \
    LRESULT ret = (x);                                \
    if((ret) != S_OK)                                 \
    {                                                 \
        char buf[512];                                \
        sprintf_s(buf, 512, "- D3D11 Call Fail @%s:%d\t  Expression: %s  Code:%d \n",__FILE__,__LINE__, #x, (ret) );  \
        OutputDebugStringA(buf);                      \
    }                                                 \
} while(0)

#define MY_DEBUG_INFO OutputDebugStringA


// Safe Release Function
template <class T>
void SafeRelease(T **ppT)
{
	if (*ppT)
	{
		(*ppT)->Release(); *ppT = NULL;
	}
}

struct CB
{
	int iWidth;
	int iHeight;
};

class DX11EffectViewer
{
    struct SimpleVertex
    {
        XMFLOAT3 pos;
        XMFLOAT2 tex;
    };

public:
	DX11EffectViewer() 
		: m_pd3dDevice(NULL)
		, m_pImmediateContext(NULL)
		, m_resultImageTexture(NULL)
		, m_pVertexLayout(NULL)
		, m_imageWidth(256)
		, m_imageHeight(256)
		, m_imageSamples(256)
        , m_textureDataSize(m_imageHeight * m_imageWidth * sizeof(float) * 3)
    {
        m_ResultImage = new float[m_textureDataSize];
        clearImage();
    }

    int     imageHeight() const { return m_imageHeight; }
    int     imageWidth()  const { return m_imageWidth;  }
    void    clearImage() { ZeroMemory(m_ResultImage , m_textureDataSize * sizeof(float)); }
    

private:

	int							m_imageWidth;
	int							m_imageHeight;
	int							m_imageSamples;
	UINT						m_textureDataSize;
    float*                      m_ResultImage;

	void    SetupViewport(float topLeftX, float topLeftY, int width, int height);
	void	InitGraphics(ID3D11Device* pd3dDevice);
	void    CreateResultImageTextureAndView(ID3D11Device* pd3dDevice);
	void	UpdateTexture();
    
private:
	// Fields
	ID3D11Device*				m_pd3dDevice;
	ID3D11DeviceContext*		m_pImmediateContext;

	ID3D11Texture2D*			m_resultImageTexture;
	ID3D11ShaderResourceView*	m_resultImageTextureView;

	ID3D11VertexShader*			m_pVertexShader;
	ID3D11InputLayout*			m_pVertexLayout;
	ID3D11Buffer*				m_pVertexBuffer;
	ID3D11SamplerState*			m_pSamplerLinear;
	ID3D11PixelShader*			m_pPixelShaderResultImage;

public:

	int     initialize(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pImmediateContext);
	void	Render(ID3D11DeviceContext* pImmediateContext );

    void Destory() 
    {
	    SafeRelease(&m_resultImageTexture);
	    SafeRelease(&m_resultImageTextureView);

	    SafeRelease(&m_pVertexBuffer);
	    SafeRelease(&m_pVertexLayout);
	    SafeRelease(&m_pVertexShader);
	    SafeRelease(&m_pSamplerLinear);
	    SafeRelease(&m_pPixelShaderResultImage);
    }

};