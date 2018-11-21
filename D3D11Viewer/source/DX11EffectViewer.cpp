#include <io.h>
#include <ctime>
#include <cstdlib>
#include "DX11EffectViewer.h"

void UpdateResult(float* image, int w, int h)
{
    for (int index = 0, i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            image[index + 0] = 1.f;
            image[index + 1] = 0.f;
            image[index + 2] = 1.f;
            index += 3;
        }
    }
}

// external definition
// definition is in rt 
int update(void* data, int nx = 256, int ny = 256, int ns = 10);

int	DX11EffectViewer::initialize(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pImmediateContext)
{
    m_pd3dDevice = pd3dDevice;
    m_pImmediateContext = pImmediateContext;

    CreateResultImageTextureAndView(pd3dDevice);
	InitGraphics(pd3dDevice);
	srand((unsigned int)time(NULL));
	return 0;
}

void   DX11EffectViewer::CreateResultImageTextureAndView(ID3D11Device* pd3dDevice)
{
    if (m_resultImageTexture)m_resultImageTexture->Release(), m_resultImageTexture = NULL;

    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.Width = m_imageWidth;
    desc.Height = m_imageHeight;
    desc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
    desc.ArraySize = 1;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;

    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.MipLevels = 1;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    CHECK_D3D11_CALL( pd3dDevice->CreateTexture2D(&desc, NULL, &m_resultImageTexture));

    if (m_resultImageTextureView)m_resultImageTextureView->Release(), m_resultImageTextureView = NULL;
    D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc;
    ZeroMemory(&viewDesc, sizeof(viewDesc));
    viewDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT; // DXGI_FORMAT_R8G8B8A8_UNORM;
    viewDesc.ViewDimension = D3D_SRV_DIMENSION_TEXTURE2D;
    viewDesc.Texture2D.MipLevels = 1;
    viewDesc.Texture2D.MostDetailedMip = 0;
    CHECK_D3D11_CALL(pd3dDevice->CreateShaderResourceView(m_resultImageTexture, &viewDesc, &m_resultImageTextureView));
}

void DX11EffectViewer::UpdateTexture()
{
    static int sSamples = 0;
    if (m_imageSamples != sSamples)
    {
        update(m_ResultImage, m_imageWidth, m_imageHeight, m_imageSamples);
        sSamples += 64 ;
    }

	D3D11_MAPPED_SUBRESOURCE mappedResource;
    CHECK_D3D11_CALL(m_pImmediateContext->Map(m_resultImageTexture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource));
	memcpy(mappedResource.pData, m_ResultImage, m_textureDataSize); // copy from GPU meory into CPU memory.
	m_pImmediateContext->Unmap(m_resultImageTexture, 0);
}

void DX11EffectViewer::Render(ID3D11DeviceContext* pImmediateContext ) 
{
    assert(pImmediateContext == m_pImmediateContext);

    UpdateTexture();

	UINT offset = 0, stride = sizeof( SimpleVertex );
	m_pImmediateContext->IASetVertexBuffers( 0, 1, &m_pVertexBuffer, &stride, &offset );
	m_pImmediateContext->IASetInputLayout( m_pVertexLayout );
	m_pImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );
	m_pImmediateContext->VSSetShader( m_pVertexShader, NULL, 0 );
	m_pImmediateContext->PSSetSamplers( 0, 1, &m_pSamplerLinear );
    {
        m_pImmediateContext->PSSetShaderResources(0, 1, &m_resultImageTextureView );
        m_pImmediateContext->PSSetShader( m_pPixelShaderResultImage, NULL, 0 );
        SetupViewport(m_imageWidth + 1.f, 0.f, m_imageWidth, m_imageHeight);
        m_pImmediateContext->Draw( 4, 0 );// draw non-indexed non-instanced primitives.[vertex count, vertex offset in vertex buffer]
    }
}


/**
 *	Load full screen quad for rendering both src and dest texture.
 */
void DX11EffectViewer::InitGraphics(ID3D11Device* pd3dDevice)
{
	DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;

#if defined( DEBUG ) || defined( _DEBUG )
	dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif
	
	SimpleVertex vertices[] =
	{
		{  XMFLOAT3(-1.0f,-1.0f, 0.5f ), XMFLOAT2( 0.0f, 1.0f ) },
		{  XMFLOAT3(-1.0f, 1.0f, 0.5f ), XMFLOAT2( 0.0f, 0.0f ) },
		{  XMFLOAT3( 1.0f,-1.0f, 0.5f ), XMFLOAT2( 1.0f, 1.0f ) },
		{  XMFLOAT3( 1.0f, 1.0f, 0.5f ), XMFLOAT2( 1.0f, 0.0f ) }
	};

	D3D11_SUBRESOURCE_DATA InitData;
	ZeroMemory( &InitData, sizeof(InitData) );
	InitData.pSysMem = vertices;

	D3D11_BUFFER_DESC bd;
	ZeroMemory( &bd, sizeof(bd) );
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof( vertices);
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;
	bd.Usage     = D3D11_USAGE_DEFAULT;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER; //bind the buffer to input-assembler stage.
	bd.ByteWidth = sizeof( vertices);
    CHECK_D3D11_CALL(pd3dDevice->CreateBuffer( &bd, &InitData, &m_pVertexBuffer ));

	ID3DBlob* pErrorBlob;
	ID3DBlob* pVSBlob = NULL;
	CHECK_D3D11_CALL(D3DCompileFromFile(L"./data/fullQuad.fx", NULL, NULL, "VS", "vs_4_0", dwShaderFlags, 0, &pVSBlob, &pErrorBlob));
	if( pErrorBlob )
	{
		OutputDebugStringA( (char*)pErrorBlob->GetBufferPointer() );
		pErrorBlob->Release();
        exit(1);
	}

    CHECK_D3D11_CALL(pd3dDevice->CreateVertexShader( pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), NULL, &m_pVertexShader));

	D3D11_INPUT_ELEMENT_DESC layout[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 }
	};
    CHECK_D3D11_CALL(pd3dDevice->CreateInputLayout(layout, 2, pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), &m_pVertexLayout));
	if (pVSBlob) pVSBlob->Release();

	// Compile pixel shader
	ID3DBlob* pPSBlob = NULL;
	CHECK_D3D11_CALL( D3DCompileFromFile(L"./data/fullQuad.fx", NULL, NULL, "psSampleResultImage", "ps_4_0", dwShaderFlags, 0, &pPSBlob, &pErrorBlob) );
	if( pErrorBlob )
	{
		OutputDebugStringA( (char*)pErrorBlob->GetBufferPointer() );
		pErrorBlob->Release();
        exit(1);
	}

    CHECK_D3D11_CALL(pd3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &m_pPixelShaderResultImage));
	if (pPSBlob) pPSBlob->Release();

	// Create sampler state
	D3D11_SAMPLER_DESC sampDesc;
	ZeroMemory( &sampDesc, sizeof(sampDesc) );
	sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	sampDesc.MinLOD = 0; sampDesc.MaxLOD = D3D11_FLOAT32_MAX;
    CHECK_D3D11_CALL(pd3dDevice->CreateSamplerState( &sampDesc, &m_pSamplerLinear));

    MY_DEBUG_INFO( "- InitGraphics OK.\n" );
}


void  DX11EffectViewer::SetupViewport(float topLeftX, float topLeftY, int width, int height)
{
	D3D11_VIEWPORT vp;
	vp.Width  = (FLOAT)width;
	vp.Height = (FLOAT)height;
	vp.TopLeftX = topLeftX; vp.TopLeftY = topLeftY;
	vp.MinDepth = 0.0f; vp.MaxDepth = 1.0f;
	m_pImmediateContext->RSSetViewports( 1, &vp );
}


