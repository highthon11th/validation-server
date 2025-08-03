from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from openai import OpenAI
import PyPDF2
import io
import os
import json
import requests
import re
import base64
from PIL import Image
from typing import List
from pdf2image import convert_from_bytes
import fitz  # PyMuPDF
import asyncio
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

app = FastAPI()

# OpenAI client setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AnalysisResponse(BaseModel):
    excessive_loan: bool
    rights_restriction: bool
    trust_property: bool
    residential_use: bool
    tax_delinquency: bool
    owner_verification: bool

class LicenseRequest(BaseModel):
    address: str
    officename: str
    licensenumber: str

class LicenseResponse(BaseModel):
    verified: bool

# 행정구역 코드 매핑
CITY_CODES = {
    "서울특별시": "11", "서울시": "11", "서울": "11",
    "부산광역시": "26", "부산시": "26", "부산": "26", 
    "대구광역시": "27", "대구시": "27", "대구": "27",
    "인천광역시": "28", "인천시": "28", "인천": "28",
    "광주광역시": "29", "광주시": "29", "광주": "29",
    "대전광역시": "30", "대전시": "30", "대전": "30",
    "울산광역시": "31", "울산시": "31", "울산": "31",
    "세종특별자치시": "36", "세종시": "36", "세종": "36",
    "경기도": "41",
    "충청북도": "43", "충북": "43",
    "충청남도": "44", "충남": "44",
    "전라남도": "46", "전남": "46",
    "경상북도": "47", "경북": "47",
    "경상남도": "48", "경남": "48",
    "제주특별자치도": "50", "제주도": "50", "제주": "50",
    "강원특별자치도": "51", "강원도": "51", "강원": "51",
    "전북특별자치도": "52", "전라북도": "52", "전북": "52"
}

def parse_address(address: str) -> tuple:
    """주소에서 시, 구, 동을 추출"""
    try:
        # 시 추출
        city_code = None
        city_name = None
        for city, code in CITY_CODES.items():
            if city in address:
                city_code = code
                city_name = city
                break
        
        if not city_code:
            raise ValueError("시/도를 찾을 수 없습니다.")
        
        # 구 추출 (시 이후 부분에서)
        city_index = address.find(city_name)
        after_city = address[city_index + len(city_name):].strip()
        
        # 구 패턴 찾기
        district_pattern = r'(\S+구|\S+군|\S+시)'
        district_match = re.search(district_pattern, after_city)
        district_name = district_match.group(1) if district_match else None
        
        # 동 패턴 찾기
        dong_pattern = r'(\S+동|\S+읍|\S+면)'
        dong_match = re.search(dong_pattern, after_city)
        dong_name = dong_match.group(1) if dong_match else None
        
        return city_code, city_name, district_name, dong_name
        
    except Exception as e:
        print(f"[ERROR] 주소 파싱 실패 - address: {address}, error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"주소 파싱 실패: {str(e)}")

def get_district_codes(city_code: str) -> list:
    """시 코드로 구 목록 조회"""
    try:
        url = "https://www.vworld.kr/dtld/comm/getBeopjeongDongList.do"
        data = {
            "V_LAWD_CD": city_code,
            "GUJESI_YN": "Y"
        }
        
        response = requests.post(url, data=data)
        response.raise_for_status()
        
        result = response.json()
        return result.get("codeList", [])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"구 코드 조회 실패: {str(e)}")

def get_dong_codes(district_code: str) -> list:
    """구 코드로 동 목록 조회"""
    try:
        url = "https://www.vworld.kr/dtld/comm/getBeopjeongDongList.do"
        data = {
            "V_LAWD_CD": district_code,
            "GUJESI_YN": "Y"
        }
        
        response = requests.post(url, data=data)
        response.raise_for_status()
        
        result = response.json()
        return result.get("codeList", [])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"동 코드 조회 실패: {str(e)}")

def get_administrative_code(address: str) -> str:
    """주소로부터 행정구역 코드 획득"""
    try:
        # 주소 파싱
        city_code, city_name, district_name, dong_name = parse_address(address)
        
        # 구 코드 조회
        district_list = get_district_codes(city_code)
        district_code = None
        
        for district in district_list:
            if district_name and district_name in district["nm"]:
                district_code = district["cd"]
                break
        
        if not district_code:
            raise ValueError(f"구/군을 찾을 수 없습니다: {district_name}")
        
        # 동 코드 조회
        dong_list = get_dong_codes(district_code)
        dong_code = None
        
        for dong in dong_list:
            if dong_name and dong_name in dong["nm"]:
                dong_code = dong["cd"]
                break
        
        if not dong_code:
            raise ValueError(f"동/읍/면을 찾을 수 없습니다: {dong_name}")
        
        return dong_code
        
    except Exception as e:
        print(f"[ERROR] 행정구역 코드 획득 실패 - address: {address}, error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"행정구역 코드 획득 실패: {str(e)}")

def is_image_file(filename: str) -> bool:
    """이미지 파일인지 확인"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def is_pdf_file(filename: str) -> bool:
    """PDF 파일인지 확인"""
    return filename.lower().endswith('.pdf')

def convert_pdf_to_images(pdf_file: UploadFile) -> List[str]:
    """PDF를 이미지로 변환하여 base64 리스트로 반환"""
    try:
        pdf_bytes = pdf_file.file.read()
        
        # PDF를 이미지로 변환 (각 페이지별로)
        images = convert_from_bytes(pdf_bytes, dpi=200, fmt='PNG')
        
        base64_images = []
        for i, image in enumerate(images):
            # PIL Image를 base64로 변환
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            base64_images.append(base64_image)
        
        # Reset file pointer
        pdf_file.file.seek(0)
        return base64_images
        
    except Exception as e:
        print(f"[ERROR] PDF 이미지 변환 실패 - filename: {pdf_file.filename}, error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF 이미지 변환 실패: {str(e)}")

def upload_file_to_openai(file: UploadFile) -> str:
    """파일을 OpenAI Files API에 업로드하고 file_id 반환"""
    try:
        # 파일 내용 읽기
        file_content = file.file.read()
        
        # 이미지 파일인 경우 유효성 검사
        if is_image_file(file.filename):
            try:
                image = Image.open(io.BytesIO(file_content))
                image.verify()
            except Exception:
                raise ValueError("유효하지 않은 이미지 파일입니다.")
        
        # Reset file pointer
        file.file.seek(0)
        
        # OpenAI Files API에 업로드
        result = client.files.create(
            file=(file.filename, file_content),
            purpose="vision"
        )
        
        print(f"파일 업로드 성공 - filename: {file.filename}, file_id: {result.id}")
        return result.id
        
    except Exception as e:
        print(f"[ERROR] 파일 업로드 실패 - filename: {file.filename}, error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"파일 업로드 실패: {str(e)}")

def convert_pdf_to_file_ids(pdf_file: UploadFile) -> List[str]:
    """PDF를 이미지로 변환하고 OpenAI에 업로드하여 file_id 리스트 반환"""
    try:
        pdf_bytes = pdf_file.file.read()
        
        # PDF를 이미지로 변환 (각 페이지별로)
        images = convert_from_bytes(pdf_bytes, dpi=200, fmt='PNG')
        
        file_ids = []
        for i, image in enumerate(images):
            # PIL Image를 바이트로 변환
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            
            # OpenAI Files API에 업로드
            result = client.files.create(
                file=(f"{pdf_file.filename}_page_{i+1}.png", image_bytes),
                purpose="vision"
            )
            file_ids.append(result.id)
            print(f"PDF 페이지 업로드 성공 - page: {i+1}, file_id: {result.id}")
        
        # Reset file pointer
        pdf_file.file.seek(0)
        return file_ids
        
    except Exception as e:
        print(f"[ERROR] PDF 이미지 변환 실패 - filename: {pdf_file.filename}, error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF 이미지 변환 실패: {str(e)}")

def analyze_files_with_openai(file_contents: List[dict]) -> dict:
    """OpenAI API를 사용하여 파일들 분석 (Files API 사용)"""
    
    # input 내용 구성
    input_content = [
        {
            "type": "input_text", 
            "text": """
**중요**: 제공된 문서들의 텍스트 내용을 정확히 읽고 분석해주세요. 반드시 문서에 명시된 구체적인 근거를 바탕으로만 판단하세요.

다음 6가지 항목을 문서 내용을 근거로 정확히 판단해주세요:

**1. 과다한 대출 여부 (excessive_loan)**
- 등기부등본의 "을구" 또는 "채무/근저당" 섹션을 확인
- 근저당권 설정액, 채권최고액 등이 과도하게 높은지 판단
- 채무 내역이 다수 기재되어 있는지 확인
- 근거: 문서에 기재된 구체적인 금액, "채무 금액" 이 없다면 false로 판단

**2. 권리제한사항 여부 (rights_restriction)**
- 등기부등본의 "갑구" 섹션을 확인
- "압류", "가압류", "처분금지", "가처분" 등의 기재사항 확인
- 근거: 문서에 명시된 구체적인 제한사항 텍스트

**3. 신탁 여부 (trust_property)**
- 등기부등본에서 "신탁" 관련 기재사항 확인
- 소유권이 신탁회사나 신탁은행으로 되어있는지 확인
- 근거: 문서에 기재된 신탁 관련 명시적 텍스트

**4. 주택용도 여부 (residential_use)**
- 건축물대장의 "용도" 또는 "건물용도" 항목 확인
- "단독주택", "공동주택", "아파트", "연립주택", "다세대주택" 등이면 true
- "상가", "사무소", "근린생활시설", "공장", "창고" 등이면 false
- 근거: 건축물대장에 명시된 정확한 용도 텍스트

**5. 체납세금 여부 (tax_delinquency)**
- 납세증명서의 "체납액" 또는 "미납세액" 항목 확인
- 체납금액이 0원이 아니거나 "없음"이 아닌 경우 확인
- 근거: 문서에 기재된 구체적인 체납 금액

**6. 등기부등본 소유자와 납세증명서 성명 일치 여부 (owner_verification)**
- 납세증명서의 "성명" 또는 "납세자명"
- 등기부등본의 "소유자" 이름
- 세 문서의 성명이 "정확히 일치"하는지 확인
- 근거: 각 문서에 기재된 구체적인 성명

**분석 지침:**
- 문서에 해당 정보가 명확히 기재되지 않은 경우에는 긍정으로 판단
- 추측하지 말고 오직 문서의 텍스트 내용만을 근거로 판단
- 금액, 성명, 용도 등은 정확한 텍스트 매칭으로 판단

**중요** : 6번을 특히 잘 확인해야합니다. "소유자 성명이 1곳이라도 다른경우"에도 false로 판단해야합니다. 예를들어 나머지는 모두 소유주 명이 동일하지만 한곳이라도 나머지와 다른경우에는 false입니다.

반드시 다음 JSON 형식으로만 답변하세요:

{
  "excessive_loan": true or false,
  "rights_restriction": true or false,
  "trust_property": true or false,
  "residential_use": true or false,
  "tax_delinquency": true or false,
  "owner_verification": true or false
}
"""
        }
    ]
    
    # 파일 내용 추가 (file_id 방식)
    for i, file_content in enumerate(file_contents):
        if file_content["type"] == "file_id":
            input_content.append({
                "type": "input_image",
                "file_id": file_content["content"]
            })
        elif file_content["type"] == "pdf_file_ids":
            # PDF에서 변환된 여러 이미지들의 file_id 추가
            for j, file_id in enumerate(file_content["content"]):
                input_content.append({
                    "type": "input_image",
                    "file_id": file_id
                })

    def make_openai_request():
        """OpenAI API 요청을 수행하는 함수"""
        response = client.responses.create(
            model="chatgpt-4o-latest",  # 최신 GPT-4 모델 사용
            input=[{
                "role": "user",
                "content": input_content
            }]
        )
        return response

    try:
        print("OpenAI API 요청 시작...")
        
        # ThreadPoolExecutor를 사용하여 타임아웃 처리
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(make_openai_request)
            try:
                # 90초 타임아웃 설정
                response = future.result(timeout=90)
                print("OpenAI API 응답 수신 완료")
                
                # 응답 내용 파싱
                response_text = response.output_text
                print(f"OpenAI 응답: {response_text}")  # 디버깅용 출력
                
                # JSON 파싱 시도
                parsed_result = parse_openai_response(response_text)
                if parsed_result:
                    return parsed_result
                else:
                    print("JSON 파싱 실패, 기본값 반환")
                    return get_default_analysis_result()
                    
            except TimeoutError:
                print("[WARNING] OpenAI API 요청 타임아웃 (90초), 기본값 반환")
                return get_default_analysis_result()

    except Exception as e:
        print(f"[ERROR] OpenAI API 오류: {str(e)}, 기본값 반환")
        return get_default_analysis_result()

def parse_openai_response(response_text: str) -> dict:
    """OpenAI 응답에서 JSON을 파싱하는 함수"""
    try:
        # JSON 블록을 찾기 위한 패턴들
        import re
        
        # ```json 블록 찾기
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
        else:
            # 단순 JSON 객체 찾기
            json_pattern = r'\{[^{}]*"excessive_loan"[^{}]*\}'
            match = re.search(json_pattern, response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                # 전체 텍스트에서 JSON 파싱 시도
                json_str = response_text.strip()
        
        # JSON 파싱
        result = json.loads(json_str)
        
        # 필요한 키들이 모두 있는지 확인
        required_keys = ["excessive_loan", "rights_restriction", "trust_property", 
                        "residential_use", "tax_delinquency", "owner_verification"]
        
        if all(key in result for key in required_keys):
            # boolean 값으로 변환
            for key in required_keys:
                if isinstance(result[key], str):
                    result[key] = result[key].lower() in ['true', '1', 'yes']
            return result
        else:
            print(f"필수 키 누락: {result}")
            return None
            
    except Exception as e:
        print(f"JSON 파싱 오류: {str(e)}")
        return None

def get_default_analysis_result() -> dict:
    """타임아웃 또는 오류 시 반환할 기본 분석 결과"""
    return {
        "excessive_loan": False,  # 안전하게 false로 설정
        "rights_restriction": False,  # 안전하게 false로 설정
        "trust_property": False,  # 안전하게 false로 설정
        "residential_use": True,  # 주택용도는 안전하게 true로 설정
        "tax_delinquency": False,  # 안전하게 false로 설정
        "owner_verification": True  # 안전하게 true로 설정
    }

@app.post("/api/analyze_house", response_model=AnalysisResponse)
async def analyze_house(
    files: List[UploadFile] = File(..., description="분석할 문서 파일들 (PDF 또는 이미지): 납세증명서, 등기부등본, 건축물대장 등")
):
    """임차주택 안전성 분석 API (파일 기반 분석)"""
    
    if not files:
        print(f"[ERROR] 파일 없음 - files: {files}")
        raise HTTPException(status_code=400, detail="최소 1개 이상의 파일이 필요합니다.")
    
    # 파일 검증 및 처리
    file_contents = []
    
    for file in files:
        if not file.filename:
            print(f"[ERROR] 파일명 없음 - file: {file}")
            raise HTTPException(status_code=400, detail="파일명이 없는 파일이 있습니다.")
        
        # 파일 타입 확인
        if is_pdf_file(file.filename):
            try:
                pdf_file_ids = convert_pdf_to_file_ids(file)
                file_contents.append({
                    "type": "pdf_file_ids",
                    "filename": file.filename,
                    "content": pdf_file_ids
                })
            except Exception as e:
                print(f"[ERROR] PDF 처리 실패 - filename: {file.filename}, error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"PDF 처리 실패 ({file.filename}): {str(e)}")
                
        elif is_image_file(file.filename):
            try:
                file_id = upload_file_to_openai(file)
                file_contents.append({
                    "type": "file_id",
                    "filename": file.filename,
                    "content": file_id
                })
            except Exception as e:
                print(f"[ERROR] 이미지 처리 실패 - filename: {file.filename}, error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"이미지 처리 실패 ({file.filename}): {str(e)}")
                
        else:
            print(f"[ERROR] 지원되지 않는 파일 형식 - filename: {file.filename}")
            raise HTTPException(
                status_code=400, 
                detail=f"지원되지 않는 파일 형식입니다: {file.filename}. PDF 또는 이미지 파일만 업로드 가능합니다."
            )
    
    if not file_contents:
        print(f"[ERROR] 처리 가능한 파일 없음 - file_contents: {file_contents}")
        raise HTTPException(status_code=400, detail="처리 가능한 파일이 없습니다.")
    
    try:
        # OpenAI로 분석
        analysis_result = analyze_files_with_openai(file_contents)
        
        return AnalysisResponse(**analysis_result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")

def verify_license(address: str, officename: str, licensenumber: str) -> bool:
    """중개업소 라이선스 검증"""
    try:
        # 행정구역 코드 획득
        admin_code = get_administrative_code(address)
        
        # 주소 파싱으로 각 코드 추출
        city_code, _, district_name, _ = parse_address(address)
        
        # 구 코드 조회 (시+구 코드)
        district_list = get_district_codes(city_code)
        district_code = None
        
        for district in district_list:
            if district_name and district_name in district["nm"]:
                district_code = district["cd"]
                break
        
        if not district_code:
            district_code = city_code + "00"  # 기본값
        
        # 라이선스 검증 API 호출
        url = "https://www.vworld.kr/dtld/broker/dtld_list_s001.do"
        print(city_code)
        data = {
            "v_lawd_cd": admin_code,
            "pageIndex": "1",
            "recordCountPerPage": "10",
            "v_sort": "",
            "v_sort_order": "",
            "GUJESI_YN": "Y",
            "sggCd": "",
            "raRegno": "",
            "sysRegno": "",
            "sidoCd": city_code,
            "sigunguCd": district_code,
            "dongCd": admin_code,
            "svcCode": "117",
            "v_cmp_nm": officename,
            "v_ra_regno": licensenumber,
            "v_rdealer_nm": "",
            "v_pos_gbn": ""
        }
        headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Origin': 'https://www.vworld.kr',
    'Referer': 'https://www.vworld.kr/dtld/broker/dtld_list_s001.do',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
    'Sec-Ch-Ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"macOS"',
    # 'Cookie': 'WMONID=GdD4evganN5; PJSESSIONID=CD2E6D56719E450EFD1815AE66E31FA2.potal_svr_31; SSCSID=PORTAL31&&CD2E6D56719E450EFD1815AE66E31FA2.potal_svr_31; wcs_bt=unknown:1754146253'
}
        print(data)
        response = requests.post(url, data=data, headers=headers)
        response.raise_for_status()
        
        # 응답에서 총 건수 확인
        response_text = response.text
        # print(response_text)  # 디버깅용 출력
        return "총<b>1</b>건" in response_text
        
    except Exception as e:
        print(f"[ERROR] 라이선스 검증 실패 - address: {address}, officename: {officename}, licensenumber: {licensenumber}, error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"라이선스 검증 실패: {str(e)}")

@app.post("/api/get_license", response_model=LicenseResponse)
async def get_license(request: LicenseRequest):
    """중개업소 라이선스 검증 API"""
    try:
        verified = verify_license(request.address, request.officename, request.licensenumber)
        return LicenseResponse(verified=verified)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검증 중 오류 발생: {str(e)}")

@app.get("/")
async def root():
    return {"message": "House Analysis API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)