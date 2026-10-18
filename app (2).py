"""Hispanic Heritage Historical Data Explorer — self-contained Streamlit app."""
from __future__ import annotations
import ast
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from html import escape
from xml.sax.saxutils import escape as xml_escape
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from scipy.optimize import brentq
from sklearn.metrics import mean_absolute_error, r2_score
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

COUNTRIES = {'Mexico':'MX','Brazil':'BR','Argentina':'AR','Colombia':'CO','Chile':'CL','Peru':'PE','Venezuela':'VE','Ecuador':'EC','Uruguay':'UY'}
# name: (official code, unit, definition, lower bound, upper bound)
INDICATORS = {
 'Life expectancy':('SP.DYN.LE00.IN','years','Expected years of life for a newborn if current age-specific mortality rates persisted.',0,150),
 'Unemployment (national estimate)':('SL.UEM.TOTL.NE.ZS','% of total labor force','Share of the labor force without work, available for and seeking work; national estimates may differ in methods.',0,100),
 'GDP per capita':('NY.GDP.PCAP.CD','current US$ per person','Gross domestic product divided by midyear population; economic output per person, not average wealth or household income.',0,None),
 'Crude birth rate':('SP.DYN.CBRT.IN','births per 1,000 people','Live births during the year per 1,000 midyear population; a rate, not a birth count.',0,1000),
 'Net migration':('SM.POP.NETM','people','Immigrants minus emigrants, including citizens and noncitizens; negative means net outward migration, not total emigration.',None,None),
 'Intentional homicide rate':('VC.IHR.PSRC.P5','homicides per 100,000 people','Victims of unlawful deaths purposefully inflicted by another person per 100,000 population.',0,None),
 'Consumer price inflation':('FP.CPI.TOTL.ZG','annual %','Annual percentage change in the consumer price index; negative values represent deflation.',-100,None),
 'Central government debt':('GC.DOD.TOTL.GD.ZS','% of GDP','Central government debt stock as a percentage of GDP; coverage and accounting may vary.',0,None),
 'Total reserves including gold':('FI.RES.TOTL.CD','current US$','Monetary gold, special drawing rights, IMF reserve position, and foreign exchange under monetary authority control.',0,None),
 'External balance on goods/services':('NE.RSB.GNFS.ZS','% of GDP','Exports of goods and services minus imports as a share of GDP; this is not the current-account balance.',None,None),
}
LIMITS = 'Polynomial fits describe associations, not causes. High-degree extrapolation can diverge rapidly. MAE and residuals summarize in-sample error, not a 95% prediction interval. Revisions, missing years, national definitions, inflation and exchange rates can affect comparisons. Edited data are hypothetical scenarios. No historical cause is inferred.'
X = sp.Symbol('x', real=True)

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_data(country, code, start, end):
    """Return original annual records (including missing values) and provenance."""
    url = f'https://api.worldbank.org/v2/country/{country}/indicator/{code}'
    params = {'date':f'{start}:{end}','format':'json','per_page':1000}
    response = requests.get(url, params=params, timeout=(8,30))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload,list) or len(payload)<2 or not isinstance(payload[0],dict):
        raise ValueError('World Bank returned an error or an unexpected response.')
    if not payload[1] or not isinstance(payload[1],list):
        raise ValueError('No annual records were returned for this selection.')
    if int(payload[0].get('pages',1))>1:
        raise ValueError('Unexpected pagination: refusing an incomplete response.')
    rows=[]
    for item in payload[1]:
        year=int(item['date'])
        if start<=year<=end:
            rows.append({'year':year,'value':item.get('value')})
    data=pd.DataFrame(rows,columns=['year','value']).sort_values('year').reset_index(drop=True)
    if data.empty: raise ValueError('No records in the requested window.')
    if data.year.duplicated().any(): raise ValueError('Duplicate source years; source needs review.')
    data['value']=pd.to_numeric(data.value,errors='raise')
    return data, {'api_url':response.url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                  'source_last_updated':payload[0].get('lastupdated','not supplied')}

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_metadata(code):
    response=requests.get(f'https://api.worldbank.org/v2/indicator/{code}',params={'format':'json'},timeout=(8,20))
    response.raise_for_status()
    p=response.json()
    if not isinstance(p,list) or len(p)<2 or not p[1]: raise ValueError('Metadata unavailable.')
    i=p[1][0]
    return {'official_name':i.get('name',''),'source_note':i.get('sourceNote',''),'source_organization':i.get('sourceOrganization','')}

def validate_table(table, indicator):
    if table.empty: raise ValueError('Enter at least one complete year/value row.')
    d=table[['year','value']].copy()
    for c in d:
        d[c]=pd.to_numeric(d[c],errors='coerce')
        if not np.isfinite(d[c].to_numpy(dtype=float)).all(): raise ValueError(f'{c}: every cell must be a finite number; remove incomplete rows.')
    if ((d.year%1)!=0).any(): raise ValueError('Years must be integers.')
    if not d.year.between(1950,2100).all(): raise ValueError('Years must be between 1950 and 2100.')
    if d.year.duplicated().any(): raise ValueError('Duplicate years are not allowed. Keep one value per year.')
    lo,hi=INDICATORS[indicator][3:]
    if lo is not None and (d.value<lo).any(): raise ValueError(f'{indicator}: values must be at least {lo}.')
    if hi is not None and (d.value>hi).any(): raise ValueError(f'{indicator}: values must be at most {hi}.')
    d.year=d.year.astype(int)
    return d.sort_values('year').reset_index(drop=True)

def fit_model(data, degree):
    if len(data)<degree+1: raise ValueError(f'Degree {degree} needs at least {degree+1} distinct years; there are {len(data)}.')
    base=float(data.year.mean())
    # Polynomial.fit scales centered time internally. convert() returns coefficients
    # in ordinary t powers; the SAME resulting Polynomial predicts everywhere.
    p=np.polynomial.Polynomial.fit(data.year.to_numpy()-base,data.value.to_numpy(),degree).convert()
    fitted=p(data.year.to_numpy()-base)
    return {'p':p,'base':base,'degree':degree,'r2':float(r2_score(data.value,fitted)),
            'mae':float(mean_absolute_error(data.value,fitted)),'fitted':fitted}

def predict(model, years):
    return np.asarray(model['p'](np.asarray(years,dtype=float)-model['base']),dtype=float)

def equation(model):
    return 'y = '+' + '.join(f'({c:.16g})'+('' if k==0 else f'*t^{k}') for k,c in enumerate(model['p'].coef))

def parse_years(text):
    good=[]; errors=[]
    for item in text.split(','):
        item=item.strip()
        try:
            if not item: raise ValueError()
            y=int(item)
            if not 1950<=y<=2100: raise ValueError()
            good.append(y)
        except ValueError: errors.append(item or '(empty item)')
    return sorted(set(good)),errors

def results_table(years,model,data,indicator):
    years=sorted(set(map(int,years)))
    values=predict(model,years)
    lo,hi=INDICATORS[indicator][3:]
    actual=dict(zip(data.year,data.value))
    rows=[]
    for y,v in zip(years,values):
        flag=not np.isfinite(v) or (lo is not None and v<lo) or (hi is not None and v>hi)
        rows.append({'year':y,'estimate':v,'units':INDICATORS[indicator][1],
                     'classification':'within fitted range' if data.year.min()<=y<=data.year.max() else 'outside fitted range',
                     'observed_working_value':actual.get(y,np.nan),
                     'value_type':'fitted value at observed year' if y in actual else 'model estimate; no observation at this year',
                     'constraint_warning':'implausible / outside stated constraint' if flag else '',
                     'extrapolation_warning':'unvalidated polynomial extrapolation' if y<data.year.min() or y>data.year.max() else ''})
    return pd.DataFrame(rows)

def make_chart(data,model,indicator,horizon,increment,results,comparisons):
    fig,ax=plt.subplots(figsize=(10,4.5))
    low,high=int(data.year.min()),int(data.year.max())
    dense=np.linspace(low,high,700)
    ax.scatter(data.year,data.value,s=18,label='Working observations',color='#176b68')
    ax.plot(dense,predict(model,dense),label='In-range polynomial',color='#23395b')
    ax.axvline(high,color='gray',linestyle=':',label='Last fitted year')
    ax.axvline(low,color='gray',linestyle=':')
    if horizon:
        forward=np.linspace(high,high+horizon,400)
        ax.plot(forward,predict(model,forward),'--',color='#c25228',label='Unvalidated extrapolation')
    samples=np.arange(low,high+horizon+1,increment)
    ax.scatter(samples,predict(model,samples),marker='|',s=40,color='#ad7696',label=f'Model samples every {increment} yr')
    if not results.empty: ax.scatter(results.year,results.estimate,marker='*',s=90,color='#d49b14',label='Calculated estimates')
    for name,table in comparisons.items():
        # Keep null rows: Matplotlib leaves gaps rather than implying observations.
        ax.plot(table.year,table.value,marker='.',alpha=.7,label=f'{name}: source observations')
    ax.set(xlabel='Calendar year',ylabel=f'{indicator} ({INDICATORS[indicator][1]})')
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best');fig.tight_layout()
    return fig

def export_excel(summary,working,original,results,samples,comparison):
    b=io.BytesIO()
    with pd.ExcelWriter(b,engine='xlsxwriter',engine_kwargs={'options':{'strings_to_formulas':False,'strings_to_urls':False}}) as writer:
        for name,table in [('Summary',pd.DataFrame(list(summary.items()),columns=['field','value'])),('Working data',working),('Original source',original),('Calculated years',results),('Model samples',samples),('Comparisons',comparison)]:
            table.to_excel(writer,sheet_name=name,index=False)
            writer.sheets[name].set_column(0,max(0,len(table.columns)-1),23)
    return b.getvalue()

def export_pdf(summary,working,results,figure):
    b=io.BytesIO();styles=getSampleStyleSheet();story=[]
    # Matplotlib ships this font: no separate local font file is required.
    from matplotlib import font_manager
    if 'HeritageSans' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont('HeritageSans',font_manager.findfont('DejaVu Sans')))
    for style in styles.byName.values(): style.fontName='HeritageSans'
    def para(text,style='BodyText'):
        story.append(Paragraph(xml_escape(str(text)),styles[style]));story.append(Spacer(1,6))
    para('Hispanic Heritage Historical Data Explorer','Title')
    for k,v in summary.items(): para(f'{k}: {v}')
    pic=io.BytesIO();figure.savefig(pic,format='png',dpi=140);pic.seek(0)
    story.append(Image(pic,width=6.8*inch,height=3.06*inch))
    story.append(KeepTogether([Paragraph('Findings and limitations',styles['Heading2']),
                              Paragraph(xml_escape(LIMITS),styles['BodyText']),Spacer(1,6)]))
    def table(title,df):
        para(title,'Heading2')
        if df.empty: para('No calculated rows.');return
        # Paragraph cells wrap long classifications and repeat headers over pages.
        rows=[[Paragraph(xml_escape(str(c)),styles['BodyText']) for c in df.columns]]
        rows += [[Paragraph(xml_escape(f'{v:.7g}' if isinstance(v,(float,np.floating)) else str(v)),styles['BodyText']) for v in row] for row in df.itertuples(index=False,name=None)]
        t=Table(rows,repeatRows=1,colWidths=[6.8*inch/len(df.columns)]*len(df.columns))
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3eeee')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.2,colors.lightgrey)]));story.append(t)
    table('Working-data preview (first 12 rows)',working.head(12))
    result_columns=[c for c in ['year','estimate','units','classification','constraint_warning'] if c in results]
    table('All calculated results',results[result_columns].rename(columns={'year':'Year','estimate':'Estimate','units':'Units','classification':'Range status','constraint_warning':'Warning'}))
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('HeritageSans',8)
        canvas.drawString(36,16,'Hispanic Heritage Data Lab')
        canvas.drawRightString(576,16,f'Page {doc.page}');canvas.restoreState()
    SimpleDocTemplate(b,pagesize=(612,792),leftMargin=36,rightMargin=36,topMargin=30,bottomMargin=30).build(story,onFirstPage=footer,onLaterPages=footer)
    return b.getvalue()

# No eval(), sympify(untrusted_text), or arbitrary Python execution.
def parse_function(text):
    text=text.strip().replace('^','**').replace('π','pi')
    if len(text)>180: raise ValueError('Use at most 180 characters.')
    funcs={'sin':sp.sin,'cos':sp.cos,'tan':sp.tan,'exp':sp.exp,'log':sp.log,'ln':sp.log,'sqrt':sp.sqrt}
    def visit(node):
        if isinstance(node,ast.Expression): return visit(node.body)
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            if not math.isfinite(node.value) or abs(node.value)>1e8: raise ValueError('Constants must be finite and no larger than 100 million.')
            return sp.Rational(str(node.value))
        if isinstance(node,ast.Name) and node.id in ('x','pi','e'): return {'x':X,'pi':sp.pi,'e':sp.E}[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)): return -visit(node.operand) if isinstance(node.op,ast.USub) else visit(node.operand)
        if isinstance(node,ast.BinOp):
            a,b=visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add): return sp.Add(a,b,evaluate=False)
            if isinstance(node.op,ast.Sub): return sp.Add(a,sp.Mul(-1,b,evaluate=False),evaluate=False)
            if isinstance(node.op,ast.Mult): return sp.Mul(a,b,evaluate=False)
            if isinstance(node.op,ast.Div):
                if b==0: raise ValueError('Division by zero.')
                return sp.Mul(a,sp.Pow(b,-1,evaluate=False),evaluate=False)
            if isinstance(node.op,ast.Pow):
                if b.free_symbols or not b.is_Rational or abs(b)>8 or b.q>2: raise ValueError('Powers must be constants from -8 to 8, integers or halves. Use exp(k*x) for exponentials.')
                return sp.Pow(a,b,evaluate=False)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in funcs and len(node.args)==1 and not node.keywords:
            return funcs[node.func.id](visit(node.args[0]),evaluate=False)
        raise ValueError('Use x, numbers, + - * / ^, parentheses, pi, e, sin, cos, tan, exp, ln/log or sqrt. Write 2*x, not 2x.')
    try:
        tree=ast.parse(text,mode='eval')
        if sum(1 for _ in ast.walk(tree))>90: raise ValueError('Expression too complex; simplify it.')
        expr=visit(tree)
        if expr.has(sp.zoo,sp.nan,sp.oo,-sp.oo): raise ValueError('Function is undefined.')
        return expr
    except (SyntaxError,TypeError,RecursionError) as exc: raise ValueError('Invalid notation. Example: 3*x^2 + 2*x - 1.') from exc

def evaluate(expr,values):
    a=np.asarray(values,dtype=float)
    with np.errstate(all='ignore'):
        try:
            raw=sp.lambdify(X,expr,modules='numpy')(a)
            z=np.broadcast_to(np.asarray(raw,dtype=complex),a.shape).copy()
            result=np.where(np.abs(z.imag)<1e-9,z.real,np.nan).astype(float)
            return np.where(np.isfinite(result),result,np.nan)
        except (ValueError,TypeError,ZeroDivisionError,OverflowError,NameError): return np.full(a.shape,np.nan)

def value_at(expr,x): return float(evaluate(expr,np.array([x]))[0])

def domain_for(expr,a,b):
    try:
        domain=sp.calculus.util.continuous_domain(expr,X,sp.S.Reals)
        return domain,domain.intersect(sp.Interval(a,b)),None
    except (NotImplementedError,ValueError,TypeError):
        return None,None,'Symbolic domain could not be resolved. Numerical domain checks are incomplete; inspect the graph and formula.'

def domain_segments(domain,a,b):
    if domain is None: return [(a,b)]
    window=sp.Interval(a,b)
    # Resolve periodic excluded sets within the finite window first (e.g. tan poles).
    if isinstance(domain,sp.Complement):
        inter=domain.args[0].intersect(window)-domain.args[1].intersect(window)
    else: inter=domain.intersect(window)
    parts=list(inter.args) if isinstance(inter,sp.Union) else [inter]
    return [(float(p.start),float(p.end)) for p in parts if isinstance(p,sp.Interval)]

def checked_slope(expr,derivative,point):
    """Reject visible corners/undefined one-sided neighborhoods before a tangent."""
    v=value_at(expr,point);slope=value_at(derivative,point)
    h=1e-6*max(1,abs(point))
    left=(v-value_at(expr,point-h))/h
    right=(value_at(expr,point+h)-v)/h
    if not np.isfinite([v,slope,left,right]).all(): return np.nan
    if abs(left-right)>1e-3*max(1,abs(left),abs(right)): return np.nan
    return slope

def numerical_roots(expr,a,b,domain):
    """Polynomial roots or sign-bracketed numerical roots; never claim exhaustive general roots."""
    if expr==0: return [],'Derivative is identically zero on its domain.'
    roots=[]
    if expr.is_polynomial(X):
        coeff=np.array([float(c) for c in sp.Poly(sp.expand(expr.doit()),X).all_coeffs()])
        for z in np.roots(coeff):
            if abs(z.imag)<1e-6 and a<z.real<b: roots.append(float(z.real))
        method='Numerical polynomial roots (including repeated roots within numerical tolerance).'
    else:
        for l,r in domain_segments(domain,a,b):
            eps=max((r-l)*1e-8,1e-8)
            if r-l<=2*eps: continue
            grid=np.linspace(l+eps,r-eps,1601);vals=evaluate(expr,grid)
            for i in range(len(grid)-1):
                if not np.isfinite(vals[i:i+2]).all(): continue
                if abs(vals[i])<1e-10: roots.append(float(grid[i]))
                if vals[i]*vals[i+1]<0:
                    try:
                        root=brentq(lambda z:value_at(expr,z),grid[i],grid[i+1])
                        if abs(value_at(expr,root))<1e-5: roots.append(root)
                    except (ValueError,RuntimeError): pass
        method='Numerical sign-bracket scan; may miss touching roots, closely spaced roots or rapid oscillations.'
    roots=sorted(roots);unique=[]
    for z in roots:
        if not unique or abs(z-unique[-1])>1e-5: unique.append(z)
    return unique,method

def analysis_points(expr,a,b):
    d1=sp.diff(expr,X);d2=sp.diff(d1,X)
    domain,restricted,error=domain_for(expr,a,b)
    roots,method=numerical_roots(d1,a,b,domain)
    second,method2=numerical_roots(d2,a,b,domain)
    # Domain boundaries of derivatives supply nondifferentiable candidates, e.g. sqrt(x^2).
    ddom,_,_=domain_for(d1,a,b)
    candidates=list(roots)
    if domain is not None and ddom is not None:
        lost=domain.intersect(sp.Interval.open(a,b))-ddom
        if isinstance(lost,sp.FiniteSet): candidates += [float(z) for z in lost if z.is_real]
    def sides(der,z,all_points):
        boundaries=[v for pair in domain_segments(domain,a,b) for v in pair]
        distances=[abs(z-w) for w in [a,b]+all_points+boundaries if abs(z-w)>1e-7]
        h=min([1e-3*max(1,b-a)]+[v/4 for v in distances])
        return value_at(der,z-h),value_at(der,z+h)
    critical=[]
    for z in sorted(set(candidates)):
        if not np.isfinite(value_at(expr,z)): continue
        left,right=sides(d1,z,candidates)
        kind='stationary / unclassified; no verified extremum'
        if np.isfinite(left) and np.isfinite(right):
            if left>0 and right<0: kind='local maximum (f\' changes + to -)'
            elif left<0 and right>0: kind='local minimum (f\' changes - to +)'
        critical.append({'x (approx)':z,'f(x)':value_at(expr,z),'classification':kind})
    infl=[]
    for z in second:
        left,right=sides(d2,z,second)
        if np.isfinite(value_at(expr,z)) and np.isfinite([left,right]).all() and left*right<0:
            infl.append({'x (approx)':z,'f(x)':value_at(expr,z),'confirmation':'f\'\' changes sign numerically'})
    return d1,d2,domain,restricted,error,pd.DataFrame(critical),pd.DataFrame(infl),method+' '+method2

def behavior_intervals(der,points,domain,a,b,label_positive,label_negative):
    bounds=sorted(set([a,b]+points+[v for pair in domain_segments(domain,a,b) for v in pair]));rows=[]
    for l,r in zip(bounds,bounds[1:]):
        if r-l<1e-8: continue
        mid=(l+r)/2
        if domain is not None and domain.contains(mid)!=sp.S.true: continue
        val=value_at(der,mid)
        if np.isfinite(val): rows.append({'interval (approx)':f'({l:.7g}, {r:.7g})','sample sign':label_positive if val>1e-9 else label_negative if val< -1e-9 else 'zero / unresolved'})
    return pd.DataFrame(rows)

CONTEXTS={
 'Business':('50 + 12*x - x^2','time','months','profit','thousand US$','Business indicator'),
 'Engineering':('2*x^2','time','s','position','m','Moving object / projectile'),
 'Medical':('10*x*exp(-0.5*x)','time since dose','hours','drug concentration','mg/L','Tank / drug concentration'),
 'Sports science':('3*x + 0.2*sin(2*x)','time','s','position','m','Walking person with prosthetic leg'),
 'Biology':('100*exp(0.2*x)','time','days','population','cells','Population / cell colony'),
 'Bioengineering':('20*(1-exp(-0.4*x))','time','hours','tissue mass','g','Population / cell colony'),
 'Economic forecasting':('100 + 2*x + 0.1*x^2','time from baseline','years','output index','index points','Business indicator'),
 'Weather':('20 + 5*sin(pi*x/12)','time','hours','temperature','degrees C','Thermometer / weather'),
 'Custom':('x^3 - 3*x','input','input units','response','output units','Moving object / projectile')}
VISUALS=['Walking person with prosthetic leg','Moving object / projectile','Tank / drug concentration','Population / cell colony','Thermometer / weather','Business indicator']

def simulation(visual,expr,point,a,b,yname,yunit):
    ys=evaluate(expr,np.linspace(a,b,500));valid=ys[np.isfinite(ys)];v=value_at(expr,point)
    st.markdown(f'**{visual}**')
    if not len(valid) or not np.isfinite(v): st.warning('Simulation paused at an undefined value.');return
    low,high=np.percentile(valid,[5,95]);q=float(np.clip((v-low)/(high-low),0,1)) if high>low else .5
    st.caption(f'Schematic scale: 5th–95th percentiles of finite sampled f(x): {low:.4g} to {high:.4g} {yunit}. Visual position/fill saturates; numerical values are never clipped. This is an illustration, not a physical/medical prediction.')
    n=int(100*q);position=45+int(410*q)
    if visual=='Walking person with prosthetic leg':
        phase=math.sin(point*5)*13
        drawing=f'<g transform="translate({position},0)"><circle cx="0" cy="48" r="12" fill="#226c69"/><path d="M0 61 L0 104 M0 73 L-20 89 M0 73 L20 85 M0 104 L{-13+phase:.1f} 140" stroke="#226c69" stroke-width="7" fill="none"/><path d="M0 104 L17 121 L{22-phase:.1f} 140" stroke="#bd7922" stroke-width="7" fill="none"/></g>'
    elif visual=='Moving object / projectile': drawing=f'<circle cx="{position}" cy="85" r="18" fill="#226c69"/><path d="M30 145 H490" stroke="#777"/>'
    elif visual=='Population / cell colony': drawing=''.join(f'<circle cx="{35+(i%20)*23}" cy="{40+(i//20)*24}" r="7" fill="#226c69"/>' for i in range(1+n))
    elif visual=='Thermometer / weather': drawing=f'<rect x="240" y="20" width="24" height="120" rx="10" fill="#eee"/><rect x="245" y="{140-n}" width="14" height="{n}" fill="#bb513c"/><circle cx="252" cy="145" r="18" fill="#bb513c"/>'
    else: drawing=f'<rect x="50" y="40" width="420" height="90" fill="#eee"/><rect x="50" y="40" width="{420*q}" height="90" fill="#226c69"/>'
    st.markdown(f'<svg viewBox="0 0 520 175" role="img" aria-label="{escape(visual)}" style="width:100%;max-height:210px">{drawing}</svg>',unsafe_allow_html=True)
    st.write(f'{yname} = {v:.7g} {yunit} at x = {point:.5g}.')

def history_ui(course,printer):
    st.header('Historical data explorer')
    research=st.text_input('Research question','How has life expectancy changed in Mexico, and how does it compare with Chile?')
    c1,c2=st.columns(2)
    country=c1.selectbox('Featured country',list(COUNTRIES))
    indicator=c2.selectbox('Indicator',list(INDICATORS))
    code,unit,definition,*_=INDICATORS[indicator]
    st.info('Brazil is a Portuguese-speaking Latin American comparison country. It is not a Spanish-speaking Hispanic country.')
    st.write(definition);st.caption(f'Units: {unit} | Indicator: {code}')
    st.markdown(f'[World Bank indicator](https://data.worldbank.org/indicator/{code})')
    current=datetime.now(timezone.utc).year;start,end=current-70,current-1
    st.caption(f'Requested window: {start}–{end}, inclusive (70 calendar years). Actual coverage depends on published data.')
    if st.button('Refresh source (clear cached requests)'): fetch_data.clear();fetch_metadata.clear();st.rerun()
    try:
        with st.spinner('Retrieving official World Bank records…'): original,provenance=fetch_data(COUNTRIES[country],code,start,end)
    except (requests.RequestException,ValueError,KeyError,TypeError) as exc:
        st.error(f'Source retrieval failed: {exc}. No synthetic data were substituted. Retry or use the independent function lab below.');return
    source=original.dropna().reset_index(drop=True)
    st.write(f'{len(source)} nonmissing observations; {len(original)-len(source)} returned missing values; {70-len(original)} unreturned calendar years.')
    if source.empty: st.warning('No nonmissing observations. Choose another indicator.');return
    st.write(f'Actual source coverage: {int(source.year.min())}–{int(source.year.max())}.')
    st.caption(f"Retrieved UTC: {provenance['retrieved_utc']} | World Bank update: {provenance['source_last_updated']}")
    st.markdown(f"[Exact API request]({provenance['api_url']})")
    with st.expander('Official metadata and original source table'):
        try: st.json(fetch_metadata(code))
        except (requests.RequestException,ValueError,KeyError,TypeError): st.warning('Live metadata unavailable; the concise definition above remains labeled by indicator code.')
        st.dataframe(original,hide_index=True)
    identity=hashlib.sha256((country+code+original.to_json()).encode()).hexdigest()
    if st.session_state.get('source_identity')!=identity:
        st.session_state.source_identity=identity;st.session_state.working=source.copy();st.session_state.editor_generation=st.session_state.get('editor_generation',0)+1
    if st.button('Reset to source'):
        st.session_state.working=source.copy();st.session_state.editor_generation+=1;st.rerun()
    st.caption('Edit complete year/value rows. Missing source rows are retained only in the original table. Invalid edits block modeling until corrected.')
    edited=st.data_editor(st.session_state.working,num_rows='dynamic',hide_index=True,key=f'editor_{st.session_state.editor_generation}')
    try: data=validate_table(edited,indicator)
    except ValueError as exc: st.error(str(exc));return
    modified=not data.equals(source)
    status='USER-MODIFIED SCENARIO' if modified else 'Unmodified World Bank observations'
    st.warning(status) if modified else st.success(status)
    c1,c2,c3=st.columns(3)
    degree=c1.slider('Polynomial degree',3,8,3)
    increment=c2.slider('Model sample increment (years)',1,10,1)
    horizon=c3.slider('Forward display horizon (years)',0,50,10)
    selected=st.multiselect('Compare source observations (same indicator)',[x for x in COUNTRIES if x!=country])
    comparisons={};comparison_rows=[]
    for name in selected:
        try:
            comp,meta=fetch_data(COUNTRIES[name],code,start,end);comparisons[name]=comp
            available=comp.dropna()
            st.caption(f'{name}: {len(available)} observations; coverage '+(f'{int(available.year.min())}–{int(available.year.max())}' if len(available) else 'none')+f"; retrieved {meta['retrieved_utc']}. Gaps are not interpolated.")
            comparison_rows.append(comp.assign(country=name,api_url=meta['api_url'],retrieved_utc=meta['retrieved_utc']))
        except (requests.RequestException,ValueError,KeyError,TypeError) as exc: st.warning(f'{name}: comparison unavailable: {exc}')
    try: model=fit_model(data,degree)
    except ValueError as exc: st.error(str(exc));return
    signature=hashlib.sha256((identity+str(degree)+data.to_json()).encode()).hexdigest()
    if st.session_state.get('model_signature')!=signature:
        st.session_state.model_signature=signature;st.session_state.single_years=[];st.session_state.batch_years=[]
    st.session_state.history_model=(model,data.copy(),country,indicator,status)
    st.code(equation(model)+f"\nt = year - {model['base']:.16g}",language='text')
    st.caption('Coefficients use centered time t; full float precision is shown. Predictions use the stored polynomial, not the displayed rounded text.')
    c1,c2,c3=st.columns(3)
    c1.metric('In-sample R²',f"{model['r2']:.6f}");c2.metric('In-sample MAE',f"{model['mae']:.6g} {unit}");c3.metric('Working observations',len(data))
    st.write(f'Fitted historical range: {int(data.year.min())}–{int(data.year.max())}; observed mean: {data.value.mean():.7g} {unit}. Observed value range: {data.value.min():.6g} to {data.value.max():.6g} {unit}.')
    holdout='Not enough data for chronological holdout (need degree+1 training rows and at least 3 test rows).'
    test_count=max(3,int(math.ceil(len(data)*.2)))
    if len(data)-test_count>=degree+1:
        train,test=data.iloc[:-test_count],data.iloc[-test_count:];hm=fit_model(train,degree);hp=predict(hm,test.year)
        holdout=f'Chronological holdout: train {int(train.year.min())}–{int(train.year.max())}; test {int(test.year.min())}–{int(test.year.max())}, n={len(test)}; test MAE={mean_absolute_error(test.value,hp):.7g} {unit}; test R²={r2_score(test.value,hp):.7g}. Final displayed model refits all working observations.'
    st.info(holdout)
    st.subheader('Calculate model values')
    year=st.number_input('Single year',1950,2100,2025,1)
    if st.button('Calculate Value'): st.session_state.single_years=[int(year)]
    mode=st.radio('Batch input',['Year range','Custom years'],horizontal=True)
    if mode=='Year range':
        c1,c2,c3=st.columns(3);ys=c1.number_input('First batch year',1950,2100,2000);ye=c2.number_input('Last batch year',1950,2100,2030);step=c3.slider('Batch step (years)',1,10,1)
        requested=list(range(int(ys),int(ye)+1,step)) if ys<=ye else [];errors=[] if ys<=ye else ['First year must not exceed last year.']
    else:
        text=st.text_input('Comma-separated years','2000, 2010, 2020, 2030');requested,errors=parse_years(text)
    st.caption(f'Valid requested years: {requested}')
    if errors: st.warning('Invalid input: '+', '.join(errors)+'. Valid custom entries can still be calculated.')
    if st.button('Calculate batch'): st.session_state.batch_years=requested
    single=results_table(st.session_state.single_years,model,data,indicator)
    batch=results_table(st.session_state.batch_years,model,data,indicator)
    if not single.empty: st.write('Saved single-year result');st.dataframe(single,hide_index=True)
    if not batch.empty: st.write('Saved batch results');st.dataframe(batch,hide_index=True)
    results=results_table(st.session_state.single_years+st.session_state.batch_years,model,data,indicator)
    results['data_status']=status
    if not results.empty and (results.constraint_warning.ne('').any() or results.extrapolation_warning.ne('').any()): st.warning('Some requested values are extrapolations or violate plausible constraints. They are displayed without clipping and are not validated forecasts.')
    samples=results_table(np.arange(int(data.year.min()),int(data.year.max())+horizon+1,increment),model,data,indicator)
    if not samples.empty and samples.constraint_warning.ne('').any(): st.warning('Displayed polynomial samples violate indicator constraints. Reduce the horizon or reconsider the model.')
    fig=make_chart(data,model,indicator,horizon,increment,results,comparisons);st.pyplot(fig)
    if not printer:
        with st.expander('Model sample table and residuals'):
            st.dataframe(samples,hide_index=True)
            residual=data.assign(fitted=model['fitted'],residual=data.value-model['fitted']);st.dataframe(residual,hide_index=True)
            rf,ax=plt.subplots(figsize=(9,2.4));ax.axhline(0,color='gray');ax.scatter(data.year,residual.residual);ax.set(xlabel='Year',ylabel=f'Residual ({unit})');st.pyplot(rf);plt.close(rf)
    st.write(LIMITS)
    st.write('Domain: the fitted polynomial accepts every real input, but data support only the fitted historical interval. Its mathematical range and end behavior need not be physically meaningful.')
    leading=model['p'].coef[-1]
    st.write(f'Degree {degree}; leading coefficient {leading:.6g}. As t → +∞, the polynomial tends to {"+∞" if leading>0 else "−∞"}; as t → −∞, it tends to {"+∞" if leading*((-1)**degree)>0 else "−∞"}.')
    st.caption('Use the Function Lab with “Historical best fit” to inspect average rates, turning behavior, and course-specific calculus analysis in centered time.')
    summary={'Research question':research,'Country':country,'Indicator':indicator,'Code':code,'Units':unit,'Definition':definition,'Data status':status,
             **provenance,'Requested window':f'{start}–{end}','Source count':len(source),'Working count':len(data),'Fitted range':f'{int(data.year.min())}–{int(data.year.max())}',
             'Observed mean':float(data.value.mean()),'Observed minimum':float(data.value.min()),'Observed maximum':float(data.value.max()),
             'Descriptive finding':f'Working values changed from {data.value.iloc[0]:.7g} in {int(data.year.iloc[0])} to {data.value.iloc[-1]:.7g} in {int(data.year.iloc[-1])} {unit}. This endpoint comparison does not establish a cause or a monotonic trend.',
             'Degree':degree,'Base year':model['base'],'Equation':equation(model),'In-sample R2':model['r2'],'In-sample MAE':model['mae'],'Holdout':holdout,
             'Sample increment':increment,'Forward horizon':horizon,'Comparison countries':', '.join(selected),'Limitations':LIMITS}
    st.subheader('Report and downloads')
    st.write(f'{research} | {country} | {indicator} | {status}')
    st.caption('Exports describe the historical model. Function Lab expressions are independent classroom simulations and are not included as World Bank data.')
    csv=data.assign(fitted=model['fitted'],residual=data.value-model['fitted'],country=country,indicator=indicator,units=unit,data_status=status,degree=degree,base_year=model['base'],api_url=provenance['api_url'],retrieved_utc=provenance['retrieved_utc'],research_question=research,equation=equation(model),in_sample_r2=model['r2'],in_sample_mae=model['mae'])
    # One CSV includes observation rows and calculated rows with an explicit record type.
    csv['record_type']='working observation';calc=results.copy();calc['record_type']='calculated estimate'
    csv=pd.concat([csv,calc],ignore_index=True).fillna('')
    a,b,c=st.columns(3)
    a.download_button('Download CSV',csv.to_csv(index=False).encode('utf-8'),'heritage_report.csv','text/csv')
    b.download_button('Download Excel (.xlsx)',export_excel(summary,data,original,results,samples,pd.concat(comparison_rows,ignore_index=True) if comparison_rows else pd.DataFrame()),'heritage_report.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    c.download_button('Download PDF',export_pdf(summary,data,results,fig),'heritage_report.pdf','application/pdf')
    plt.close(fig)


def lab_ui(course,printer):
    st.header('Function and context lab')
    st.caption('Suggested functions and parameter models are teaching examples, not measured data or medical advice. The historical best fit uses the current validated working data.')
    context=st.selectbox('Application context',list(CONTEXTS));suggested,xn,xu,yn,yu,vis=CONTEXTS[context]
    description=st.text_area('Context description',f'Explore how {yn} changes with {xn} in {context.lower()}.',key=f'desc_{context}')
    col1,col2,col3,col4=st.columns(4)
    xname=col1.text_input('Independent variable name',xn,key=f'xn_{context}');xunit=col2.text_input('Independent units',xu,key=f'xu_{context}')
    yname=col3.text_input('Dependent variable name',yn,key=f'yn_{context}');yunit=col4.text_input('Dependent units',yu,key=f'yu_{context}')
    source=st.selectbox('Function source',['Suggested function','Parameter builder','Custom expression','Historical best fit'])
    default_a,default_b=0.,10.
    if source=='Historical best fit':
        if 'history_model' not in st.session_state: st.info('Load a valid historical model first, or choose another function source.');return
        model,data,country,indicator,status=st.session_state.history_model
        expr=sum(sp.Float(float(c),17)*X**k for k,c in enumerate(model['p'].coef))
        default_a=float(data.year.min()-model['base']);default_b=float(data.year.max()-model['base'])
        st.info(f"{country}, {indicator}; {status}. x = calendar year − {model['base']:.8g}. Independent units are years; dependent units are {INDICATORS[indicator][1]}.")
        xname='centered time';xunit='years';yname=indicator;yunit=INDICATORS[indicator][1]
    else:
        if source=='Parameter builder':
            family=st.selectbox('Function family',['Polynomial','Rational','Exponential','Logarithmic','Trigonometric'])
            ca,cb,cc,cd=st.columns(4)
            A=ca.number_input('a',-1000.,1000.,1.);B=cb.number_input('b',-100.,100.,.5);C=cc.number_input('c',-1000.,1000.,0.);D=cd.number_input('d',-1000.,1000.,0.)
            if family=='Polynomial':
                power=st.slider('Builder polynomial degree',1,8,3);text=f'{A}*x^{power}+{B}*x^2+{C}*x+{D}'
            else: text={'Rational':f'{A}/(x-({B}))+({C})','Exponential':f'{A}*exp(({B})*x)+({C})','Logarithmic':f'{A}*ln(({B})*x+({C}))+({D})','Trigonometric':f'{A}*sin(({B})*x+({C}))+({D})'}[family]
            st.code(text,language='text')
        else: text=st.text_input('f(x): explicit * multiplication; ^ or ** powers',suggested,key=f'function_{context}_{source}')
        try: expr=parse_function(text)
        except ValueError as exc: st.error(str(exc));return
    col1,col2=st.columns(2)
    a=col1.number_input('Analysis interval left',-10000.,10000.,default_a,key=f'left_{source}')
    b=col2.number_input('Analysis interval right',-10000.,10000.,default_b,key=f'right_{source}')
    if a>=b: st.error('Left endpoint must be smaller than right endpoint.');return
    with st.spinner('Analyzing function and domain…'):
        try: d1,d2,domain,restricted,error,critical,inflections,method=analysis_points(expr,a,b)
        except (ValueError,TypeError,NotImplementedError,OverflowError) as exc: st.error(f'Analysis could not be completed: {exc}. Try a simpler expression or shorter interval.');return
    st.latex('f(x) = '+sp.latex(expr))
    st.write(description)
    st.write('Real continuous domain: '+str(domain) if domain is not None else 'Real domain unresolved.')
    st.caption('Domain sets use open/closed endpoints; excluded inputs include poles, logarithm restrictions and nonreal roots. The parser preserves operations so removable holes remain domain restrictions.')
    if error: st.warning(error)
    if restricted==sp.S.EmptySet: st.error('No continuous real-valued portion in this interval.');return
    if course=='Calculus':
        st.latex("f'(x) = "+sp.latex(d1));st.latex("f''(x) = "+sp.latex(d2))
        if d1.has(sp.DiracDelta) or d2.has(sp.DiracDelta):
            st.warning('This symbolic expression includes a generalized derivative term (DiracDelta). It is not an ordinary classical derivative at a corner. Numerical derivative/concavity results there are unresolved; verify one-sided limits by hand.')
    else:
        st.info('Algebra 2: inspect domain, sampled range, end behavior and average rate. Precalculus also uses changes in average rates to discuss turning and concavity. Select Calculus for the full derivative analysis.')
    fingerprint=hashlib.sha256((str(expr)+str(a)+str(b)+context).encode()).hexdigest()[:12]
    point_key='point_'+fingerprint
    if point_key not in st.session_state: st.session_state[point_key]=float(a)
    playing_key='playing_'+fingerprint
    speed=st.slider('Playback speed (interval % per tick)',1,20,3)
    visual=st.selectbox('Contextual visual',VISUALS,index=VISUALS.index(vis))
    show1=show2=False
    if course=='Calculus':
        c1,c2=st.columns(2);show1=c1.checkbox("Show f' curve",True);show2=c2.checkbox("Show f'' curve",False)
    show0=st.checkbox('Show f curve',True)
    show_tangent=st.checkbox('Show tangent line',course=='Calculus') if course=='Calculus' else False
    show_secant=st.checkbox('Show secant line',True)
    sec=st.slider('Average-rate interval endpoints',float(a),float(b),(float(a),float(b)),key='sec_'+fingerprint)
    # A fragment reruns just the synchronized panel. No blocking loop or sleep.
    @st.fragment(run_every=.3)
    def moving_panel():
        c1,c2,c3=st.columns(3)
        if c1.button('Play',key='play_'+fingerprint): st.session_state[playing_key]=True
        if c2.button('Pause',key='pause_'+fingerprint): st.session_state[playing_key]=False
        if c3.button('Reset playback',key='reset_'+fingerprint): st.session_state[playing_key]=False;st.session_state[point_key]=float(a)
        if st.session_state.get(playing_key,False):
            new=st.session_state[point_key]+(b-a)*speed/100
            st.session_state[point_key]=float(a if new>b else new)
        point=st.slider('Current input x',float(a),float(b),key=point_key)
        fv=value_at(expr,point);dv=checked_slope(expr,d1,point);ddv=value_at(d2,point)
        ua,ub=sec;va,vb=value_at(expr,ua),value_at(expr,ub)
        rate=(vb-va)/(ub-ua) if ub>ua and np.isfinite([va,vb]).all() else np.nan
        st.write(f'x = {point:.7g} {xunit}; {yname} = {fv:.7g} {yunit}.')
        st.write(f'Average rate on [{ua:.5g}, {ub:.5g}]: {rate:.7g} {yunit}/{xunit}.' if np.isfinite(rate) else 'Average rate undefined: choose distinct endpoints with finite values.')
        if course=='Calculus':
            st.write(f"Instantaneous rate f′ = {dv:.7g} {yunit}/{xunit}; rate of change of the rate f″ = {ddv:.7g} {yunit}/{xunit}².")
            if yname.lower()=='position': st.write('Here f′ is velocity and f″ is acceleration. Their signs describe direction and the change in velocity.')
            else: st.write(f"At this input, {yname} is {'increasing' if dv>0 else 'decreasing' if dv<0 else 'stationary or undefined'}; its slope is {'increasing' if ddv>0 else 'decreasing' if ddv<0 else 'unchanging or undefined'}. These are mathematical statements about the chosen model.")
            if np.isfinite([fv,dv]).all(): st.latex(f'y = ({fv:.8g}) + ({dv:.8g})(x-({point:.8g}))')
        fig=go.Figure()
        segments=domain_segments(domain,a,b)
        for idx,(l,r) in enumerate(segments):
            eps=max((r-l)*1e-7,1e-8);grid=np.linspace(l+eps,r-eps,600) if r-l>eps*2 else np.array([])
            for form,label,enabled,color in [(expr,'f',show0,'#1f7771'),(d1,"f′",show1,'#c07a28'),(d2,"f″",show2,'#916fa8')]:
                if not enabled: continue
                vals=evaluate(form,grid)
                # Symbolic sign terms arise at corners such as sqrt(x^2).
                # Break their plotted jumps instead of drawing a connecting line.
                for atom in form.atoms(sp.sign):
                    arg=atom.args[0]
                    if arg.is_polynomial(X):
                        jumps,_=numerical_roots(arg,l,r,domain)
                        for jump in jumps:
                            nearest=int(np.argmin(np.abs(grid-jump))) if len(grid) else 0
                            vals[max(0,nearest-1):nearest+2]=np.nan
                # Extra gap mask for unresolved singularities or numerical blowup.
                if len(vals)>1:
                    delta=np.abs(np.diff(vals));scale=np.nanmedian(np.abs(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else 1
                    bad=np.where(delta>max(1,scale)*100)[0];vals[bad]=np.nan
                fig.add_trace(go.Scatter(x=grid,y=vals,name=label,mode='lines',connectgaps=False,showlegend=idx==0,line={'color':color}))
        if np.isfinite(fv): fig.add_trace(go.Scatter(x=[point],y=[fv],name='Current point',mode='markers',marker={'size':12,'color':'#cc492f'}))
        if show_tangent and np.isfinite([fv,dv]).all():
            xx=np.array([max(a,point-(b-a)/8),min(b,point+(b-a)/8)])
            fig.add_trace(go.Scatter(x=xx,y=fv+dv*(xx-point),name='Tangent',line={'dash':'dot'}))
        if show_secant and np.isfinite(rate): fig.add_trace(go.Scatter(x=[ua,ub],y=[va,vb],name='Secant',line={'dash':'dash'}))
        fig.update_layout(xaxis_title=f'{xname} ({xunit})',yaxis_title=f'f ({yunit}); derivative units shown above',height=460,template='plotly_white')
        st.plotly_chart(fig,width='stretch',key='graph_'+fingerprint)
        simulation(visual,expr,point,a,b,yname,yunit)
    moving_panel()
    sample_x=np.linspace(a,b,1001);sample_y=evaluate(expr,sample_x);finite=sample_y[np.isfinite(sample_y)]
    if len(finite): st.write(f'Sampled range on selected interval: approximately {finite.min():.7g} to {finite.max():.7g} {yunit}. This is not a proof of the mathematical range or absolute extrema.')
    if expr.is_polynomial(X):
        poly=sp.Poly(sp.expand(expr.doit()),X)
        if poly.degree()>0:
            st.write(f'Polynomial degree {poly.degree()}, leading coefficient {poly.LC()}. Changing degree changes possible turns; changing coefficients changes scale, direction and locations. Opposite end directions for odd degree; same end directions for even degree, with sign determined by the leading coefficient.')
            lead=float(poly.LC());deg=int(poly.degree())
            st.write(f'End behavior: as x → +∞, f(x) → {"+∞" if lead>0 else "−∞"}; as x → −∞, f(x) → {"+∞" if lead*(-1)**deg>0 else "−∞"}. This mathematical behavior may leave the context’s physically meaningful range.')
        else: st.write('Constant function: same value and zero rate everywhere in its domain.')
    if course in ('Precalculus','Calculus'):
        edges=np.linspace(a,b,9);values=evaluate(expr,edges);rates=np.diff(values)/np.diff(edges)
        st.dataframe(pd.DataFrame({'left':edges[:-1],'right':edges[1:],'average rate':rates,'change in average rate':np.r_[np.nan,np.diff(rates)]}),hide_index=True)
        st.caption('Increasing average rates suggest concave-up behavior; decreasing rates suggest concave-down behavior. Coarse interval samples alone cannot prove turns or inflections.')
    if course=='Calculus':
        st.subheader('Critical points, monotonicity and concavity')
        st.caption('Function and derivative formulas are symbolic (exact for rational inputs, approximate for fitted float coefficients). ALL listed locations/classifications are numerical, tolerance-based and may be incomplete. '+method)
        st.dataframe(critical if not critical.empty else pd.DataFrame({'result':['No isolated critical point detected. A constant function has every interior point stationary.']}),hide_index=True)
        roots,_=numerical_roots(d1,a,b,domain);second,_=numerical_roots(d2,a,b,domain)
        st.write('Increasing/decreasing intervals (midpoint sign tests between detected candidates):')
        st.dataframe(behavior_intervals(d1,roots,domain,a,b,'increasing','decreasing'),hide_index=True)
        st.write('Concavity intervals (midpoint sign tests between detected candidates):')
        st.dataframe(behavior_intervals(d2,second,domain,a,b,'concave up','concave down'),hide_index=True)
        st.write('Verified numerical inflection candidates:')
        st.dataframe(inflections if not inflections.empty else pd.DataFrame({'result':['No sign-changing second-derivative zero detected.']}),hide_index=True)
        closed=domain is not None and sp.Interval(a,b).is_subset(domain)==True
        if closed:
            candidates=[a,b]+(critical['x (approx)'].tolist() if not critical.empty else [])
            extrema=pd.DataFrame({'x (approx)':candidates,'f(x)':[value_at(expr,z) for z in candidates]})
            extrema=extrema[np.isfinite(extrema['f(x)'])]
            if not extrema.empty:
                st.write('Absolute extrema candidates, INCLUDING endpoints:');st.dataframe(extrema,hide_index=True)
                st.write(f"Candidate minimum {extrema['f(x)'].min():.8g}; candidate maximum {extrema['f(x)'].max():.8g}. For continuous polynomials, all stationary roots are sought; general functions need a completeness proof before calling these absolute extrema.")
        else: st.warning('The function is not established continuous across this closed interval; the endpoint-plus-critical-point theorem does not establish absolute extrema here. Split at domain restrictions or choose a valid closed interval.')
        st.caption('A zero derivative alone is not an extremum. An inflection requires continuity and a concavity change. Numerical scans may miss nondifferentiable points or even-multiplicity roots in nonpolynomial functions; unresolved results are not a symbolic proof.')


def main():
    st.set_page_config(page_title='Hispanic Heritage Historical Data Explorer',page_icon='🌎',layout='wide')
    st.markdown('<style>.stApp {background: #fbfcfa;} h1,h2,h3 {color:#205d59;} @media print { [data-testid="stSidebar"], button, [data-testid="stSlider"], [data-testid="stNumberInput"], [data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stTextInput"], [data-testid="stDataEditor"] {display:none!important;} .stApp {background:white;} }</style>',unsafe_allow_html=True)
    st.title('Hispanic Heritage Historical Data Explorer')
    st.caption('Hispanic Heritage Data Lab · authentic observations, transparent models, changeable contexts')
    course=st.sidebar.selectbox('Mathematics pathway',['Honors Algebra 2','Precalculus','Calculus'],index=2)
    printer=st.sidebar.checkbox('Printer-friendly report view')
    if printer:
        st.info('Printer-friendly view: historical report only, static chart, simplified controls. PDF provides a paginated printable report. Turn this off to return to the function lab.')
        history_ui(course,printer)
    else:
        t1,t2,t3=st.tabs(['Historical data','Function & calculus lab','How to defend the work'])
        with t1:
            # Clear stale bridge before retrieval: failed/invalid source cannot feed a previous fit.
            st.session_state.pop('history_model',None)
            history_ui(course,printer)
        with t2: lab_ui(course,printer)
        with t3:
            st.markdown('''1. **Source → validation:** World Bank records are cached without imputation. Nulls remain in the original table; complete working rows are validated before fitting. Edits change only the scenario.
2. **Validation → fit:** center years at their mean. Fit a degree 3–8 polynomial with a numerically scaled least-squares routine; convert once to a polynomial in centered time. Single, batch, graph and export predictions call the same stored polynomial.
3. **Evidence → interpretation:** in-sample R²/MAE describe fit; chronological holdout measures later-year performance of a separately trained model. Neither establishes causality or reliable distant forecasts.
4. **Function lab:** a restricted syntax tree creates a SymPy expression, then symbolic derivatives. Domain-aware segments avoid connecting across known discontinuities. Numerical sign tests classify candidates; incomplete scans are explicitly labeled.
5. **Interaction:** one fragment updates the point, tangent, values, explanations and schematic together. Playback is a sequence of fragment reruns, not a blocking loop.
6. **Exports:** pandas creates CSV, XlsxWriter creates a real workbook, and ReportLab embeds a Matplotlib chart and tables in a real PDF. No credentials or additional local modules are needed.''')
            st.write(LIMITS)

if __name__=='__main__': main()
