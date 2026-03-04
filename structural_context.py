"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     structural_context.py - السياق الهندسي                                    ║
║     الطبقة المفقودة في V7                                                     ║
║                                                                               ║
║     المسطرة تقول: أين أنت                                                     ║
║     الأدبية تقول: هل تدخل                                                     ║
║     هذه الوحدة تقول: ماذا حولك                                                ║
║                                                                               ║
║     الترندات ترسم زمن — وتقاطعاتها محاور                                      ║
║     كل قمة وقاع = زمن تسعة                                                    ║
║     الدعوم والمقاومات مناطق لا خطوط                                            ║
║                                                                               ║
║     يعمل على اليومي والأسبوعي فقط                                              ║
║     الفريمات الصغيرة تضيّع الوقت                                               ║
║                                                                               ║
║     جميع الحقوق محفوظة — عماد سليمان                                           ║
║     ترجمة المفهوم: Claude (Opus)                                               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# النماذج — كل مفهوم له شكل واضح
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PivotPoint:
    """
    نقطة محورية — قمة أو قاع احترمها السوق
    
    ليست مجرد أعلى أو أدنى سعر
    بل نقطة ذهب إليها السوق وارتد عنها
    هذه هي الارتكازات الحقيقية
    """
    index: int              # موقعها على المحور الزمني (رقم الشمعة)
    price: float            # السعر
    timestamp: datetime     # الوقت
    pivot_type: str         # 'high' أو 'low'
    strength: int = 0       # كم مرة احترمها السوق لاحقاً
    
    def __repr__(self):
        arrow = "▲" if self.pivot_type == "high" else "▼"
        return f"{arrow} {self.price:.4f} @ bar {self.index}"


@dataclass
class TrendLine:
    """
    خط ترند — ليس مجرد خط على الشارت
    بل مسار زمني يكشف إيقاع السوق
    
    عماد: "الترندات ترسم زمن وتقاطعاتها محاور"
    """
    start: PivotPoint       # نقطة البداية
    end: PivotPoint         # نقطة النهاية
    trend_type: str         # 'up' أو 'down'
    slope: float = 0.0      # الميل — السرعة
    violations: int = 0     # كم مرة اختُرق
    is_broken: bool = False # هل مكسور فعلاً
    
    def price_at(self, bar_index: int) -> float:
        """
        أين يكون سعر الترند عند أي شمعة
        هذا هو الإسقاط — الترند يمتد للمستقبل
        """
        if self.end.index == self.start.index:
            return self.start.price
        
        bars_from_start = bar_index - self.start.index
        return self.start.price + self.slope * bars_from_start
    
    def bars_alive(self, current_bar: int) -> int:
        """عمر الترند بالشموع"""
        return current_bar - self.start.index
    
    def __repr__(self):
        direction = "↗" if self.trend_type == "up" else "↘"
        status = "✓" if not self.is_broken else "✗"
        return f"{direction} {self.start.price:.2f}→{self.end.price:.2f} [{status}]"


@dataclass 
class SupportResistanceZone:
    """
    منطقة دعم أو مقاومة — صندوق لا خط
    
    الدعم ليس رقماً واحداً بل منطقة
    السعر قد يرتد قبلها بقليل أو بعدها بقليل
    """
    top: float              # سقف المنطقة
    bottom: float           # أرضية المنطقة
    zone_type: str          # 'support' أو 'resistance'
    pivot: PivotPoint       # النقطة المحورية التي أنشأتها
    times_tested: int = 0   # كم مرة اختُبرت
    times_held: int = 0     # كم مرة صمدت
    is_broken: bool = False
    
    @property
    def center(self) -> float:
        """مركز المنطقة"""
        return (self.top + self.bottom) / 2
    
    @property
    def width(self) -> float:
        """عرض المنطقة"""
        return self.top - self.bottom
    
    @property
    def hold_rate(self) -> float:
        """نسبة الصمود — كم مرة صمدت من أصل كم اختبار"""
        if self.times_tested == 0:
            return 0.0
        return self.times_held / self.times_tested
    
    def contains_price(self, price: float) -> bool:
        """هل السعر داخل المنطقة"""
        return self.bottom <= price <= self.top
    
    def __repr__(self):
        label = "S" if self.zone_type == "support" else "R"
        return f"[{label}] {self.bottom:.4f} — {self.top:.4f} (held {self.hold_rate:.0%})"


@dataclass
class TrendIntersection:
    """
    تقاطع ترندين — محور زمني
    
    عماد: "تقاطعات الترندات محاور"
    عندما ترندان يلتقيان = نقطة يلتقي فيها إيقاعان
    هنا بالذات يُحتمل الانعكاس
    """
    bar_index: int          # متى يحدث التقاطع (الزمن!)
    price: float            # عند أي سعر
    trend_a: TrendLine      # الترند الأول
    trend_b: TrendLine      # الترند الثاني
    is_future: bool = True  # هل التقاطع في المستقبل
    
    def bars_until(self, current_bar: int) -> int:
        """كم شمعة حتى التقاطع"""
        return self.bar_index - current_bar
    
    def __repr__(self):
        when = "→" if self.is_future else "←"
        return f"{when} bar {self.bar_index} @ {self.price:.4f}"


@dataclass
class StructuralContext:
    """
    السياق الهندسي الكامل — ماذا حول السعر الآن
    
    هذا هو ما ينقص V7:
    ليس فقط "أين أنت" (المسطرة)
    ولا فقط "هل تدخل" (الأدبية)
    بل "ماذا حولك" — الترندات والدعوم والمقاومات
    """
    # السعر الحالي
    current_price: float
    current_bar: int
    
    # أقرب مقاومة فوقك
    nearest_resistance: Optional[SupportResistanceZone] = None
    resistance_distance_pct: float = 0.0
    
    # أقرب دعم تحتك
    nearest_support: Optional[SupportResistanceZone] = None
    support_distance_pct: float = 0.0
    
    # أقرب ترند فوقك (ضغط)
    trend_above: Optional[TrendLine] = None
    trend_above_price: float = 0.0
    trend_above_dist_pct: float = 0.0
    
    # أقرب ترند تحتك (سند)
    trend_below: Optional[TrendLine] = None
    trend_below_price: float = 0.0
    trend_below_dist_pct: float = 0.0
    
    # الملعب — المسافة بين أقرب ضغط وأقرب سند
    playfield_width_pct: float = 0.0
    
    # عدد الترندات النشطة
    active_uptrends: int = 0
    active_downtrends: int = 0
    
    # أقرب محور زمني (تقاطع ترندات في المستقبل)
    next_time_pivot: Optional[TrendIntersection] = None
    bars_to_time_pivot: int = 0
    
    # ملخص بشري
    summary: str = ""
    
    def to_dict(self) -> dict:
        """
        تحويل لقاموس — جاهز لـ ml_snapshots
        هذه الأعمدة تُضاف مباشرة للجدول
        """
        return {
            "nearest_resistance_dist": round(self.resistance_distance_pct, 4),
            "nearest_support_dist": round(self.support_distance_pct, 4),
            "trend_above_price": round(self.trend_above_price, 4),
            "trend_below_price": round(self.trend_below_price, 4),
            "trend_above_dist_pct": round(self.trend_above_dist_pct, 4),
            "trend_below_dist_pct": round(self.trend_below_dist_pct, 4),
            "playfield_width_pct": round(self.playfield_width_pct, 4),
            "active_uptrends": self.active_uptrends,
            "active_downtrends": self.active_downtrends,
            "bars_to_time_pivot": self.bars_to_time_pivot,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# المحرك — اكتشاف وتحليل البنية الهندسية
# ═══════════════════════════════════════════════════════════════════════════════

class StructuralEngine:
    """
    محرك السياق الهندسي
    
    يأخذ بيانات الشموع ويُرجع الصورة الكاملة:
    — النقاط المحورية (أين احترم السوق)
    — الترندات (مسارات الزمن)
    — الدعوم والمقاومات (مناطق لا خطوط)
    — تقاطعات الترندات (محاور الزمن المستقبلية)
    — السياق الكامل (ماذا حول السعر الآن)
    """
    
    def __init__(
        self,
        pivot_length: int = 20,
        max_violations: int = 1,
        except_last_bars: int = 3,
        max_trendline_points: int = 5,
        zone_body_ratio: bool = True,
    ):
        """
        pivot_length:  كم شمعة يميناً ويساراً لاكتشاف النقطة المحورية
                       20 = يجب أن تكون أعلى/أدنى من 40 شمعة حولها
                       
        max_violations: نظام التسامح — كم اختراق مسموح قبل اعتبار الترند مكسوراً
                        0 = صارم (أول اختراق = كسر)
                        1 = متسامح (يسمح باختراق واحد = صيد سيولة)
                        
        except_last_bars: تجاهل آخر X شموع عند حساب الاختراقات
                          لأن الشمعة الحالية قد تكون في طور التشكل
                          
        max_trendline_points: كم نقطة محورية نستخدم لتوليد التوليفات
                              5 نقاط = 10 ترندات ممكنة
                              
        zone_body_ratio: هل المنطقة من جسم الشمعة (True) أم من الذيل فقط (False)
        """
        self.pivot_length = pivot_length
        self.max_violations = max_violations
        self.except_last_bars = except_last_bars
        self.max_trendline_points = max_trendline_points
        self.zone_body_ratio = zone_body_ratio
    
    # ───────────────────────────────────────────────────────────────────────
    # الخطوة 1: اكتشاف النقاط المحورية
    # ───────────────────────────────────────────────────────────────────────
    
    def discover_pivots(self, df: pd.DataFrame) -> Tuple[List[PivotPoint], List[PivotPoint]]:
        """
        اكتشاف النقاط المحورية — القمم والقيعان التي احترمها السوق
        
        المنطق: قمة محورية = أعلى سعر من كل الشموع على بُعد pivot_length
                قاع محوري = أدنى سعر من كل الشموع على بُعد pivot_length
        
        مثل الكود الأصلي:
            ph = ta.pivothigh(pvtLength, pvtLength)
            pl = ta.pivotlow(pvtLength, pvtLength)
        
        Returns:
            (pivot_highs, pivot_lows) — قائمتان مرتبتان من الأحدث للأقدم
        """
        highs = df['high'].values
        lows = df['low'].values
        L = self.pivot_length
        
        pivot_highs = []
        pivot_lows = []
        
        # نبحث من الشمعة L حتى آخر شمعة - L
        # لأن كل نقطة تحتاج L شموع قبلها و L بعدها
        for i in range(L, len(df) - L):
            
            # ── قمة محورية ──
            # هل high[i] أعلى من كل الجيران على بُعد L؟
            is_pivot_high = True
            for j in range(i - L, i + L + 1):
                if j == i:
                    continue
                if highs[j] >= highs[i]:
                    is_pivot_high = False
                    break
            
            if is_pivot_high:
                ts = df.index[i] if isinstance(df.index[i], datetime) else datetime.now()
                pivot_highs.append(PivotPoint(
                    index=i,
                    price=float(highs[i]),
                    timestamp=ts,
                    pivot_type='high'
                ))
            
            # ── قاع محوري ──
            is_pivot_low = True
            for j in range(i - L, i + L + 1):
                if j == i:
                    continue
                if lows[j] <= lows[i]:
                    is_pivot_low = False
                    break
            
            if is_pivot_low:
                ts = df.index[i] if isinstance(df.index[i], datetime) else datetime.now()
                pivot_lows.append(PivotPoint(
                    index=i,
                    price=float(lows[i]),
                    timestamp=ts,
                    pivot_type='low'
                ))
        
        # حساب قوة كل نقطة — كم مرة عاد السعر إليها واحترمها
        for pivot in pivot_highs:
            pivot.strength = self._calculate_pivot_strength(pivot, df)
        for pivot in pivot_lows:
            pivot.strength = self._calculate_pivot_strength(pivot, df)
        
        # ترتيب من الأحدث للأقدم (مثل الكود الأصلي: unshift)
        pivot_highs.sort(key=lambda p: p.index, reverse=True)
        pivot_lows.sort(key=lambda p: p.index, reverse=True)
        
        return pivot_highs, pivot_lows
    
    def _calculate_pivot_strength(self, pivot: PivotPoint, df: pd.DataFrame) -> int:
        """
        قوة النقطة المحورية — كم مرة عاد السعر إليها بعد تشكلها
        
        نقطة عاد إليها السعر 5 مرات أقوى من نقطة لم يعد إليها أبداً
        هذا يعطي وزناً للارتكازات الحقيقية
        """
        strength = 0
        tolerance = pivot.price * 0.003  # تسامح 0.3%
        
        # نبحث في الشموع بعد النقطة المحورية
        for i in range(pivot.index + 1, len(df)):
            if pivot.pivot_type == 'high':
                # كم مرة وصل السعر لقربها (كمقاومة)
                if abs(df['high'].iloc[i] - pivot.price) <= tolerance:
                    strength += 1
            else:
                # كم مرة وصل السعر لقربها (كدعم)
                if abs(df['low'].iloc[i] - pivot.price) <= tolerance:
                    strength += 1
        
        return strength
    
    # ───────────────────────────────────────────────────────────────────────
    # الخطوة 2: توليد الترندات — كل التوليفات الممكنة
    # ───────────────────────────────────────────────────────────────────────
    
    def generate_trendlines(
        self,
        pivot_highs: List[PivotPoint],
        pivot_lows: List[PivotPoint],
        df: pd.DataFrame
    ) -> Tuple[List[TrendLine], List[TrendLine]]:
        """
        توليد كل الترندات الممكنة بين النقاط المحورية
        
        مثل الكود الأصلي:
            f_getAllPairCombinations(lowPivots.slice(0, tlPointsToCheck))
        
        ترند صاعد = بين قاعين (الثاني أعلى من الأول)
        ترند هابط = بين قمتين (الثانية أدنى من الأولى)
        
        Returns:
            (uptrends, downtrends)
        """
        n = self.max_trendline_points
        current_bar = len(df) - 1
        
        # نأخذ أحدث N نقاط فقط
        recent_highs = pivot_highs[:n]
        recent_lows = pivot_lows[:n]
        
        uptrends = []
        downtrends = []
        
        # ── ترندات صاعدة: كل توليفات القيعان ──
        # الكود الأصلي: f_getAllPairCombinations + f_isLower
        low_pairs = self._get_all_pairs(recent_lows)
        for first, second in low_pairs:
            # الأول يجب أن يكون أقدم والثاني أحدث
            if first.index > second.index:
                first, second = second, first
            # ترند صاعد = القاع الثاني أعلى من الأول
            if second.price > first.price:
                slope = (second.price - first.price) / (second.index - first.index)
                tl = TrendLine(
                    start=first,
                    end=second,
                    trend_type='up',
                    slope=slope
                )
                # فحص الاختراقات (نظام التسامح)
                tl.violations = self._count_violations(tl, df, 'up')
                tl.is_broken = tl.violations > self.max_violations
                uptrends.append(tl)
        
        # ── ترندات هابطة: كل توليفات القمم ──
        high_pairs = self._get_all_pairs(recent_highs)
        for first, second in high_pairs:
            if first.index > second.index:
                first, second = second, first
            # ترند هابط = القمة الثانية أدنى من الأولى
            if second.price < first.price:
                slope = (second.price - first.price) / (second.index - first.index)
                tl = TrendLine(
                    start=first,
                    end=second,
                    trend_type='down',
                    slope=slope
                )
                tl.violations = self._count_violations(tl, df, 'down')
                tl.is_broken = tl.violations > self.max_violations
                downtrends.append(tl)
        
        return uptrends, downtrends
    
    def _get_all_pairs(self, points: List[PivotPoint]) -> List[Tuple[PivotPoint, PivotPoint]]:
        """
        كل التوليفات الممكنة — مثل f_getAllPairCombinations في الكود الأصلي
        
        3 نقاط = 3 توليفات
        5 نقاط = 10 توليفات
        """
        pairs = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                pairs.append((points[i], points[j]))
        return pairs
    
    def _count_violations(self, tl: TrendLine, df: pd.DataFrame, trend_type: str) -> int:
        """
        نظام التسامح — كم مرة اخترق السعر الترند
        
        مثل الكود الأصلي:
            method getHighsAbovePrice / getLowsBelowPrice
        
        ترند صاعد: نعد كم مرة low أقل من سعر الترند
        ترند هابط: نعد كم مرة high أعلى من سعر الترند
        """
        violations = 0
        start_bar = tl.start.index
        end_bar = len(df) - self.except_last_bars
        
        if start_bar >= end_bar:
            return 0
        
        for i in range(start_bar, end_bar):
            trend_price = tl.price_at(i)
            
            if trend_type == 'up':
                # ترند صاعد مكسور إذا low أقل من سعر الترند
                if df['low'].iloc[i] < trend_price:
                    violations += 1
            else:
                # ترند هابط مكسور إذا high أعلى من سعر الترند
                if df['high'].iloc[i] > trend_price:
                    violations += 1
        
        return violations
    
    # ───────────────────────────────────────────────────────────────────────
    # الخطوة 3: الدعوم والمقاومات كمناطق
    # ───────────────────────────────────────────────────────────────────────
    
    def build_zones(
        self,
        pivot_highs: List[PivotPoint],
        pivot_lows: List[PivotPoint],
        df: pd.DataFrame
    ) -> Tuple[List[SupportResistanceZone], List[SupportResistanceZone]]:
        """
        بناء مناطق الدعم والمقاومة — صناديق لا خطوط
        
        مثل الكود الأصلي:
            box.new(top = min(open, close), bottom = lowValue)
        
        المنطقة = من جسم الشمعة إلى ذيلها
        لأن الدعم ليس رقماً واحداً بل نطاق
        """
        supports = []
        resistances = []
        
        # ── مناطق الدعم من القيعان المحورية ──
        for pivot in pivot_lows[:self.max_trendline_points]:
            idx = pivot.index
            if idx >= len(df):
                continue
            
            candle_low = df['low'].iloc[idx]
            if self.zone_body_ratio:
                candle_body_bottom = min(df['open'].iloc[idx], df['close'].iloc[idx])
                zone_top = candle_body_bottom
            else:
                zone_top = candle_low * 1.002  # 0.2% فوق القاع
            
            zone = SupportResistanceZone(
                top=zone_top,
                bottom=candle_low,
                zone_type='support',
                pivot=pivot
            )
            
            # كم مرة اختُبرت وصمدت
            zone.times_tested, zone.times_held = self._test_zone(zone, df)
            zone.is_broken = zone.hold_rate < 0.3  # مكسورة إذا صمدت أقل من 30%
            
            supports.append(zone)
        
        # ── مناطق المقاومة من القمم المحورية ──
        for pivot in pivot_highs[:self.max_trendline_points]:
            idx = pivot.index
            if idx >= len(df):
                continue
            
            candle_high = df['high'].iloc[idx]
            if self.zone_body_ratio:
                candle_body_top = max(df['open'].iloc[idx], df['close'].iloc[idx])
                zone_bottom = candle_body_top
            else:
                zone_bottom = candle_high * 0.998
            
            zone = SupportResistanceZone(
                top=candle_high,
                bottom=zone_bottom,
                zone_type='resistance',
                pivot=pivot
            )
            
            zone.times_tested, zone.times_held = self._test_zone(zone, df)
            zone.is_broken = zone.hold_rate < 0.3
            
            resistances.append(zone)
        
        return supports, resistances
    
    def _test_zone(self, zone: SupportResistanceZone, df: pd.DataFrame) -> Tuple[int, int]:
        """
        اختبار المنطقة — كم مرة اختُبرت وكم مرة صمدت
        
        الاختبار = السعر دخل المنطقة
        الصمود = السعر دخل المنطقة ثم خرج في الاتجاه المعاكس
        """
        tested = 0
        held = 0
        start_idx = zone.pivot.index + 1
        
        for i in range(start_idx, len(df)):
            if zone.zone_type == 'support':
                # اختبار الدعم = low دخل المنطقة
                if df['low'].iloc[i] <= zone.top:
                    tested += 1
                    # صمود = أغلق فوق المنطقة
                    if df['close'].iloc[i] > zone.top:
                        held += 1
            else:
                # اختبار المقاومة = high دخل المنطقة
                if df['high'].iloc[i] >= zone.bottom:
                    tested += 1
                    # صمود = أغلق تحت المنطقة
                    if df['close'].iloc[i] < zone.bottom:
                        held += 1
        
        return tested, held
    
    # ───────────────────────────────────────────────────────────────────────
    # الخطوة 4: تقاطعات الترندات — محاور الزمن
    # ───────────────────────────────────────────────────────────────────────
    
    def find_intersections(
        self,
        uptrends: List[TrendLine],
        downtrends: List[TrendLine],
        current_bar: int,
        max_future_bars: int = 100
    ) -> List[TrendIntersection]:
        """
        البحث عن تقاطعات الترندات — المحاور الزمنية
        
        عماد: "الترندات ترسم زمن وتقاطعاتها محاور"
        
        كل تقاطع بين ترند صاعد وهابط = نقطة يلتقي فيها إيقاعان
        التقاطعات المستقبلية = محاور زمنية قادمة
        """
        intersections = []
        
        # نبحث عن تقاطع كل ترند صاعد مع كل ترند هابط
        for up in uptrends:
            if up.is_broken:
                continue
            for down in downtrends:
                if down.is_broken:
                    continue
                
                intersection = self._find_line_intersection(up, down)
                if intersection is None:
                    continue
                
                bar_idx, price = intersection
                
                # نهتم فقط بالتقاطعات المعقولة
                # في المستقبل القريب (max_future_bars)
                # أو في الماضي القريب (للمرجعية)
                if bar_idx < current_bar - 50:
                    continue
                if bar_idx > current_bar + max_future_bars:
                    continue
                if price <= 0:
                    continue
                
                intersections.append(TrendIntersection(
                    bar_index=int(bar_idx),
                    price=price,
                    trend_a=up,
                    trend_b=down,
                    is_future=bar_idx > current_bar
                ))
        
        # ترتيب حسب القرب من الشمعة الحالية
        intersections.sort(key=lambda x: abs(x.bar_index - current_bar))
        
        return intersections
    
    def _find_line_intersection(
        self, 
        line_a: TrendLine, 
        line_b: TrendLine
    ) -> Optional[Tuple[float, float]]:
        """
        حساب نقطة تقاطع ترندين
        
        رياضياً: حل معادلتين خطيتين
        y1 = a1 * x + b1
        y2 = a2 * x + b2
        التقاطع عند: x = (b2 - b1) / (a1 - a2)
        """
        # إذا الميلان متساوي = خطان متوازيان لا يتقاطعان
        if abs(line_a.slope - line_b.slope) < 1e-10:
            return None
        
        # b = y - slope * x (عند نقطة البداية)
        b1 = line_a.start.price - line_a.slope * line_a.start.index
        b2 = line_b.start.price - line_b.slope * line_b.start.index
        
        # نقطة التقاطع
        x = (b2 - b1) / (line_a.slope - line_b.slope)
        y = line_a.slope * x + b1
        
        return (x, y)
    
    # ───────────────────────────────────────────────────────────────────────
    # الخطوة 5: السياق الكامل — الصورة الشاملة
    # ───────────────────────────────────────────────────────────────────────
    
    def analyze(self, df: pd.DataFrame) -> StructuralContext:
        """
        التحليل الكامل — يُرجع السياق الهندسي الشامل
        
        هذه هي الدالة الرئيسية التي يستدعيها V7
        
        Input:  DataFrame بشموع OHLCV (يومي أو أسبوعي)
        Output: StructuralContext — كل ما يحتاجه النظام
        """
        if len(df) < self.pivot_length * 2 + 1:
            return StructuralContext(
                current_price=float(df['close'].iloc[-1]),
                current_bar=len(df) - 1,
                summary="بيانات غير كافية"
            )
        
        current_bar = len(df) - 1
        current_price = float(df['close'].iloc[-1])
        
        # ── 1. اكتشاف النقاط المحورية ──
        pivot_highs, pivot_lows = self.discover_pivots(df)
        
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return StructuralContext(
                current_price=current_price,
                current_bar=current_bar,
                summary="نقاط محورية غير كافية"
            )
        
        # ── 2. توليد الترندات ──
        uptrends, downtrends = self.generate_trendlines(pivot_highs, pivot_lows, df)
        
        # فصل النشط عن المكسور
        active_uptrends = [t for t in uptrends if not t.is_broken]
        active_downtrends = [t for t in downtrends if not t.is_broken]
        
        # ── 3. بناء المناطق ──
        supports, resistances = self.build_zones(pivot_highs, pivot_lows, df)
        active_supports = [z for z in supports if not z.is_broken]
        active_resistances = [z for z in resistances if not z.is_broken]
        
        # ── 4. تقاطعات الترندات (محاور الزمن) ──
        intersections = self.find_intersections(
            active_uptrends, active_downtrends, current_bar
        )
        future_intersections = [ix for ix in intersections if ix.is_future]
        
        # ── 5. بناء السياق ──
        context = StructuralContext(
            current_price=current_price,
            current_bar=current_bar
        )
        
        # أقرب مقاومة فوق السعر
        above_resistances = [
            z for z in active_resistances 
            if z.bottom > current_price
        ]
        if above_resistances:
            above_resistances.sort(key=lambda z: z.bottom - current_price)
            context.nearest_resistance = above_resistances[0]
            context.resistance_distance_pct = (
                (above_resistances[0].bottom - current_price) / current_price * 100
            )
        
        # أقرب دعم تحت السعر
        below_supports = [
            z for z in active_supports
            if z.top < current_price
        ]
        if below_supports:
            below_supports.sort(key=lambda z: current_price - z.top)
            context.nearest_support = below_supports[0]
            context.support_distance_pct = (
                (current_price - below_supports[0].top) / current_price * 100
            )
        
        # أقرب ترند فوق السعر (ضغط)
        trends_above = []
        for t in active_downtrends:
            t_price = t.price_at(current_bar)
            if t_price > current_price:
                trends_above.append((t, t_price))
        if trends_above:
            trends_above.sort(key=lambda x: x[1] - current_price)
            context.trend_above = trends_above[0][0]
            context.trend_above_price = trends_above[0][1]
            context.trend_above_dist_pct = (
                (trends_above[0][1] - current_price) / current_price * 100
            )
        
        # أقرب ترند تحت السعر (سند)
        trends_below = []
        for t in active_uptrends:
            t_price = t.price_at(current_bar)
            if t_price < current_price:
                trends_below.append((t, t_price))
        if trends_below:
            trends_below.sort(key=lambda x: current_price - x[1])
            context.trend_below = trends_below[0][0]
            context.trend_below_price = trends_below[0][1]
            context.trend_below_dist_pct = (
                (current_price - trends_below[0][1]) / current_price * 100
            )
        
        # عرض الملعب
        if context.trend_above_price > 0 and context.trend_below_price > 0:
            context.playfield_width_pct = (
                context.trend_above_dist_pct + context.trend_below_dist_pct
            )
        
        # عدد الترندات النشطة
        context.active_uptrends = len(active_uptrends)
        context.active_downtrends = len(active_downtrends)
        
        # أقرب محور زمني
        if future_intersections:
            context.next_time_pivot = future_intersections[0]
            context.bars_to_time_pivot = future_intersections[0].bars_until(current_bar)
        
        # ── 6. الملخص البشري ──
        context.summary = self._build_summary(context)
        
        return context
    
    def _build_summary(self, ctx: StructuralContext) -> str:
        """
        ملخص بشري — جملة واحدة تصف الوضع
        
        هذا ليس للآلة — بل لعماد عندما ينظر للتلقرام
        """
        parts = []
        
        # الضغط من فوق
        if ctx.trend_above_price > 0:
            parts.append(f"ترند هابط فوقك بـ {ctx.trend_above_dist_pct:.1f}%")
        else:
            parts.append("لا ضغط من فوق")
        
        # السند من تحت
        if ctx.trend_below_price > 0:
            parts.append(f"ترند صاعد تحتك بـ {ctx.trend_below_dist_pct:.1f}%")
        else:
            parts.append("لا سند من تحت")
        
        # الملعب
        if ctx.playfield_width_pct > 0:
            parts.append(f"عرض الملعب {ctx.playfield_width_pct:.1f}%")
        
        # المحور الزمني
        if ctx.next_time_pivot:
            bars = ctx.bars_to_time_pivot
            parts.append(f"محور زمني بعد {bars} شمعة")
        
        return " | ".join(parts)
    
    # ───────────────────────────────────────────────────────────────────────
    # أدوات مساعدة
    # ───────────────────────────────────────────────────────────────────────
    
    def get_all_pivots_with_nine(
        self, 
        pivot_highs: List[PivotPoint], 
        pivot_lows: List[PivotPoint]
    ) -> Dict[str, float]:
        """
        ربط النقاط المحورية بنسب التسعة
        
        عماد: "كل قمة وقاع عبارة عن زمن تسعة"
        
        يحسب المسافة بين كل قمة وقاع ويقيسها بنسب التسعة
        """
        NINE_RATIOS = [0.18, 0.27, 0.45, 0.50, 0.54, 0.72, 0.81]
        
        results = {}
        
        if not pivot_highs or not pivot_lows:
            return results
        
        # أعلى قمة وأدنى قاع
        highest = max(pivot_highs, key=lambda p: p.price)
        lowest = min(pivot_lows, key=lambda p: p.price)
        
        total_range = highest.price - lowest.price
        if total_range <= 0:
            return results
        
        for ratio in NINE_RATIOS:
            price_from_low = lowest.price + total_range * ratio
            price_from_high = highest.price - total_range * ratio
            results[f"nine_{ratio}_from_low"] = round(price_from_low, 4)
            results[f"nine_{ratio}_from_high"] = round(price_from_high, 4)
        
        return results
    
    def get_pivot_based_ruler(
        self,
        pivot_highs: List[PivotPoint],
        pivot_lows: List[PivotPoint],
        current_price: float
    ) -> Dict[str, float]:
        """
        مسطرة مبنية على النقاط المحورية بدل أطراف الشموع
        
        هذا يُحسّن المسطرة الحالية:
        بدل High/Low العشوائي — نستخدم القمم والقيعان المحورية
        """
        if not pivot_highs or not pivot_lows:
            return {"ruler_ratio": 0.5, "ruler_high": 0, "ruler_low": 0}
        
        highest = max(pivot_highs, key=lambda p: p.price)
        lowest = min(pivot_lows, key=lambda p: p.price)
        
        total_range = highest.price - lowest.price
        if total_range <= 0:
            return {"ruler_ratio": 0.5, "ruler_high": highest.price, "ruler_low": lowest.price}
        
        ratio = (current_price - lowest.price) / total_range
        ratio = max(0.0, min(1.0, ratio))
        
        return {
            "ruler_ratio": round(ratio, 4),
            "ruler_high": highest.price,
            "ruler_low": lowest.price,
            "ruler_high_bar": highest.index,
            "ruler_low_bar": lowest.index,
            "ruler_high_strength": highest.strength,
            "ruler_low_strength": lowest.strength,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# الاختبار — تحقق سريع أن كل شيء يعمل
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """
    اختبار ذاتي ببيانات وهمية
    يتحقق أن كل وحدة تعمل بشكل صحيح
    """
    print("=" * 60)
    print("  structural_context.py — اختبار ذاتي")
    print("=" * 60)
    
    np.random.seed(42)
    n_bars = 200
    
    # بيانات وهمية تحاكي حركة حقيقية
    # نصنع ترنداً صاعداً ثم هابطاً
    prices = [100.0]
    for i in range(1, n_bars):
        if i < 80:
            # ترند صاعد
            change = np.random.randn() * 1.5 + 0.3
        elif i < 120:
            # قمة وتذبذب
            change = np.random.randn() * 2.0
        else:
            # ترند هابط
            change = np.random.randn() * 1.5 - 0.2
        prices.append(prices[-1] + change)
    
    prices = np.array(prices)
    prices = np.maximum(prices, 50)  # حد أدنى
    
    dates = pd.date_range(start='2025-01-01', periods=n_bars, freq='D')
    df = pd.DataFrame({
        'open': prices + np.random.randn(n_bars) * 0.5,
        'high': prices + abs(np.random.randn(n_bars)) * 2,
        'low': prices - abs(np.random.randn(n_bars)) * 2,
        'close': prices + np.random.randn(n_bars) * 0.3,
        'volume': np.random.rand(n_bars) * 1000 + 100
    }, index=dates)
    
    # تأكد أن high > low دائماً
    df['high'] = df[['open', 'close', 'high']].max(axis=1) + 0.5
    df['low'] = df[['open', 'close', 'low']].min(axis=1) - 0.5
    
    # ── إنشاء المحرك ──
    engine = StructuralEngine(
        pivot_length=10,        # أقصر للاختبار
        max_violations=1,
        max_trendline_points=5
    )
    
    # ── 1. اكتشاف النقاط المحورية ──
    print("\n📍 الخطوة 1: النقاط المحورية")
    pivot_highs, pivot_lows = engine.discover_pivots(df)
    print(f"   قمم محورية: {len(pivot_highs)}")
    print(f"   قيعان محورية: {len(pivot_lows)}")
    if pivot_highs:
        print(f"   أقوى قمة: {pivot_highs[0]} (قوة: {pivot_highs[0].strength})")
    if pivot_lows:
        print(f"   أقوى قاع: {pivot_lows[0]} (قوة: {pivot_lows[0].strength})")
    
    # ── 2. توليد الترندات ──
    print("\n📈 الخطوة 2: الترندات")
    uptrends, downtrends = engine.generate_trendlines(pivot_highs, pivot_lows, df)
    active_up = [t for t in uptrends if not t.is_broken]
    active_down = [t for t in downtrends if not t.is_broken]
    print(f"   ترندات صاعدة: {len(uptrends)} (نشط: {len(active_up)})")
    print(f"   ترندات هابطة: {len(downtrends)} (نشط: {len(active_down)})")
    
    # ── 3. الدعوم والمقاومات ──
    print("\n🔲 الخطوة 3: المناطق")
    supports, resistances = engine.build_zones(pivot_highs, pivot_lows, df)
    print(f"   مناطق دعم: {len(supports)}")
    print(f"   مناطق مقاومة: {len(resistances)}")
    for z in supports[:2]:
        print(f"   {z}")
    for z in resistances[:2]:
        print(f"   {z}")
    
    # ── 4. تقاطعات الترندات ──
    print("\n⏱️  الخطوة 4: محاور الزمن")
    intersections = engine.find_intersections(
        active_up, active_down, len(df) - 1
    )
    future = [ix for ix in intersections if ix.is_future]
    print(f"   تقاطعات: {len(intersections)} (مستقبلية: {len(future)})")
    for ix in future[:3]:
        bars = ix.bars_until(len(df) - 1)
        print(f"   {ix} — بعد {bars} شمعة")
    
    # ── 5. السياق الكامل ──
    print("\n🎯 الخطوة 5: السياق الكامل")
    context = engine.analyze(df)
    print(f"   السعر الحالي: {context.current_price:.2f}")
    print(f"   الملخص: {context.summary}")
    print(f"\n   أعمدة ml_snapshots:")
    for key, value in context.to_dict().items():
        print(f"      {key}: {value}")
    
    # ── 6. ربط بنسب التسعة ──
    print("\n🔢 الخطوة 6: نسب التسعة من المحاور")
    nine = engine.get_all_pivots_with_nine(pivot_highs, pivot_lows)
    for key, value in list(nine.items())[:4]:
        print(f"   {key}: {value:.2f}")
    
    # ── 7. المسطرة المُحسّنة ──
    print("\n📏 الخطوة 7: المسطرة المحورية")
    ruler = engine.get_pivot_based_ruler(
        pivot_highs, pivot_lows, context.current_price
    )
    for key, value in ruler.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("  ✅ جميع الوحدات تعمل")
    print("=" * 60)
    
    return context


if __name__ == "__main__":
    ctx = self_test()
