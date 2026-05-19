# BLAZ4R GNIRS ITC 계산 메모

이 저장소는 BLAZ4R 대상의 Mg II 관측 시간을 산출하기 위해 작성한 코드입니다.

주요 처리 단계는 다음과 같습니다.

1. 대상의 적색편이($z$)를 기준으로 Mg II가 관측될 파장을 계산합니다.

2. UKIDSS J/H/K 측광 데이터로부터 해당 대역의 플럭스를 계산합니다.

3. 측광 필터의 중심 파장과 실제 Mg II 관측 파장의 오프셋을 고려해 색 보정(Color Correction)을 수행합니다.

4. 가정된 Mg II 선의 물리량(등가폭, 선폭)을 바탕으로 Gemini GNIRS ITC에 입력할 파라미터를 계산합니다.

5. ITC 시뮬레이션 결과로 나온 스펙트럼에서 Mg II 선폭 구간 내의 S/N(신호대잡음비) 중앙값을 구합니다.

상세한 수식 및 계산 가정은 아래 내용을 참고하시기 바랍니다.

## 파일 역할 및 구조

### `1_make_gnirs_itc_inputs.py`

* 대상 천체 목록과 UKIDSS J/H/K 측광값을 읽어옵니다.

* Mg II 관측 파장, 연속광 플럭스 밀도, Mg II 선 강도를 계산합니다.

* 계산 결과를 취합하여 `gnirs_itc_inputs.csv` 파일로 내보냅니다.

### `2_run_gnirs_itc_batch.py`

* 생성된 CSV 파라미터 테이블을 Gemini GNIRS ITC API로 전송합니다.

* 노출 횟수 30, 60, 90회 조건에 대해 각각 시뮬레이션을 돌립니다.

* ITC 서버로부터 받은 파장별 S/N 결과 데이터를 텍스트 파일로 저장합니다.

### `3_plot_gnirs_itc_snr.py`

* 저장된 파장별 S/N 스펙트럼 데이터를 읽어옵니다.

* 스펙트럼상에 Mg II 선 중심과 선폭 영역을 오버레이합니다.

* 해당 선폭 구간 내의 S/N 중앙값을 산출하고 진단용 플롯 이미지로 저장합니다.

## 상세 계산 과정 및 수식

### 1. Mg II 관측계 파장 결정

Mg II의 정지계 진공 파장($\lambda_{\text{rest}}$)은 $0.279875\,\mu\text{m}$로 고정하여 사용합니다. 적색편이($z$)를 가진 천체의 관측계 파장($\lambda_{\text{obs}}$)은 우주론적 적색편이 공식을 따릅니다.

$$
\lambda_{\text{obs}} = 0.279875 \times (1 + z)\,\mu\text{m}
$$

예를 들어, 적색편이 $z = 4.4$인 천체의 경우:

$$
\lambda_{\text{obs}} = 0.279875 \times (1 + 4.4) \approx 1.51\,\mu\text{m}
$$

이 파장은 근적외선 H-band 대역에 놓이게 됩니다.

### 2. UKIDSS 측광값의 물리 플럭스 변환

UKIDSS 데이터베이스에서 제공하는 $J, H, K$ 측광값은 Vega 등급 체계입니다. 물리량 연산을 위해 먼저 AB 등급 체계로 변환합니다. 변환에는 Hewett et al. (2006) 기준의 보정 계수를 적용합니다.

$$
m_{\text{AB}, J} = m_{\text{Vega}, J} + 0.938
$$

$$
m_{\text{AB}, H} = m_{\text{Vega}, H} + 1.379
$$

$$
m_{\text{AB}, K} = m_{\text{Vega}, K} + 1.900
$$

변환된 AB 등급($m_{\text{AB}}$)을 주파수 단위의 플럭스 밀도 $f_{\nu}$ (Jy)로 바꿉니다.

$$
f_{\nu} = 3631 \times 10^{-0.4 \times m_{\text{AB}}}\,\text{Jy}
$$

이를 다시 파장 단위의 플럭스 밀도 $f_{\lambda}$ ($\text{erg}\,\text{s}^{-1}\,\text{cm}^{-2}\,\text{\AA}^{-1}$)로 변환합니다.

$$
f_{\lambda} = \frac{f_{\nu} \cdot c}{\lambda^2}
$$

*주의: 이 수식으로 계산한 플럭스는 각각 J, H, K 필터의 중심 파장 기준이며, 아직 실제 Mg II 파장의 값은 아닙니다.*

### 3. 필터 중심 파장과 Mg II 파장 간 색 보정 (Color Correction)

근적외선 H-band 필터의 중심 파장은 대략 $1.631\,\mu\text{m}$이지만, 천체의 적색편이에 따라 실제 Mg II가 검출되는 파장은 다르게 나타납니다.

* **J132512**: $\lambda_{\text{obs, Mg II}} \approx 1.517\,\mu\text{m}$

* **J153533**: $\lambda_{\text{obs, Mg II}} \approx 1.509\,\mu\text{m}$

* **J001115**: $\lambda_{\text{obs, Mg II}} \approx 1.668\,\mu\text{m}$

필터 중심 파장에서의 플럭스를 실제 Mg II 관측 파장 위치의 연속광 플럭스로 정밀화하기 위해, Vanden Berk et al. (2001)의 SDSS 퀘이사 합성 스펙트럼 기울기를 보정에 사용합니다.

$$
f_{\nu} \propto \nu^{\alpha_{\nu}} \quad (\alpha_{\nu} = -0.44)
$$

주파수 지수 $\alpha_{\nu} = -0.44$를 적용해 파장 기준 플럭스 보정식으로 고쳐 쓰면 다음과 같습니다.

$$
f_{\lambda}(\lambda_{\text{Mg II}}) = f_{\lambda}(\lambda_{\text{filter}}) \times \left( \frac{\lambda_{\text{Mg II}}}{\lambda_{\text{filter}}} \right)^{-(\alpha_{\nu} + 2)}
$$

이 보정식의 물리적 의미는 다음과 같습니다.

* Mg II 실제 파장이 필터 중심 파장보다 단파장 영역에 치우쳐 있으면(예: J132512, J153533) 보정 후 연속광 세기가 다소 높아지며, 이에 따라 계산되는 S/N도 상승합니다.

* 반대로 중심 파장보다 장파장 영역에 있으면(예: J001115) 보정 후 연속광 세기가 깎이고 S/N도 감소합니다.

### 4. Mg II 선 세기 계산

실제 분광 자료가 없는 설계 단계이므로, 전형적인 활동성 은하핵(AGN)의 물리량을 기준으로 연산합니다.

* 정지계 등가폭 (Rest-frame Equivalent Width, $W_{\lambda, 0}$) = $30\,\text{\AA}$

* 방출선 선폭 ($\text{FWHM}$) = $4000\,\text{km/s}$

관측자가 검출하게 되는 Mg II 선 강도($F_{\text{line}}$)는 다음 수식을 거쳐 가늠합니다.

$$
F_{\text{line}} = f_{\lambda}(\lambda_{\text{Mg II}}) \times W_{\lambda, 0} \times (1 + z)
$$

우주 팽창 효과로 인해 관측계에서 늘어난 선 등가폭($W_{\lambda, \text{obs}} = W_{\lambda, 0} \times (1+z)$)을 반영하기 위해 $(1+z)$ 항을 곱합니다. 추후 선행 연구나 사전 관측 데이터가 있을 시 고유 물리량으로 대체해 연산합니다.

### 5. Gemini ITC 전송 시 유의사항

파이프라인이 생성하는 `gnirs_itc_inputs.csv` 테이블에는 계산이 이미 끝난 관측 파장($\lambda_{\text{obs}}$)이 담겨 있습니다. 따라서 ITC API로 쿼리를 보낼 때는 이중 파장 편이 적용을 피하기 위해 적색편이 변수 값을 `z = 0`으로 명시해야 합니다.

전송 단위 규격은 다음과 같이 준수합니다.

* 선 강도 ($F_{\text{line}}$): $\text{erg}\,\text{s}^{-1}\,\text{cm}^{-2}$

* 연속광 플럭스 밀도 ($f_{\lambda}$): $\text{erg}\,\text{s}^{-1}\,\text{cm}^{-2}\,\text{\AA}^{-1}$

API 플랫폼 내부 구동 시 입력 텍스트와 실제 연산 단위의 부정합이 일어날 수 있으므로 코드 내에서 유효 단위 명칭을 재검증합니다.

### 6. S/N 중앙값 산출 공식

특정 파장 단 한 곳만 표본으로 삼으면, sky emission line 노이즈나 픽셀 그리드 특이점에 의한 이상치가 계산 결과 왜곡을 일으킬 수 있습니다. 따라서 방출선 속도 프로파일 구간 전체의 S/N을 대표할 수 있는 중앙값을 구합니다.

먼저 FWHM 속도 폭($4000\,\text{km/s}$)을 파장 단위의 폭($\Delta\lambda$)으로 바꿉니다.

$$
\Delta\lambda = \lambda_{\text{obs, Mg II}} \times \frac{\text{FWHM}}{c}
$$

(단, $c$는 빛의 속도)

계산된 폭을 기준으로 다음 파장 범위 안의 S/N 배열을 슬라이싱합니다.

$$
\left[ \lambda_{\text{obs, Mg II}} - 0.5\Delta\lambda ,\; \lambda_{\text{obs, Mg II}} + 0.5\Delta\lambda \right]
$$

이 슬라이스 구간 내 데이터의 중앙값(Median)을 대표 S/N으로 활용합니다.

## 코드 실행 방법

의존성 도구 `uv`를 활용해 다음과 같이 단계별 스크립트를 실행합니다.

1. **관측 파라미터 연산 및 CSV 파일 생성**

   ```bash
   uv run python 1_make_gnirs_itc_inputs.py
   ```

2. **Gemini ITC API 일괄 구동 및 결과 수집**

   ```bash
   uv run python 2_run_gnirs_itc_batch.py
   ```

3. **결과 분석 및 S/N 진단 그래프 출력**

   ```bash
   uv run python 3_plot_gnirs_itc_snr.py
   ```

4. **단위 테스트 검증**

   ```bash
   uv run python -m unittest discover -s tests
   ```

## 시뮬레이션 해석 시 고려 사항

본 시뮬레이션에 사용된 주요 전제 조건들의 한계점은 다음과 같습니다.

- **방출선 물리량 일괄 가정**: 개별 블레이저의 중심 블랙홀 상태나 강착원반 광도에 따라 고유의 등가폭과 FWHM 변동이 발생할 수 있으나, 본 파이프라인은 초기 기준값($W_{\lambda,0} = 30\,\text{\AA}$, $\text{FWHM} = 4000\,\text{km/s}$)을 일괄 상정했습니다.
- **광대역 필터 통합에 따른 평균값**: UKIDSS 측광값은 넓은 밴드를 하나의 적분 등급으로 평균 낸 값입니다. 템플릿 색 보정을 가했더라도 미세한 방출선 구조나 관측 대역의 국소적 기울기가 실제 퀘이사 개별 스펙트럼과 정확히 부합하지 않을 수 있습니다.
