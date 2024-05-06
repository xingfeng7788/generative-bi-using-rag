import * as React from "react";
import AreaChart from "@cloudscape-design/components/area-chart";
import Box from "@cloudscape-design/components/box";
import Button from "@cloudscape-design/components/button";

const ChartPage = () => {
  return (
    <AreaChart
      series={[
        {
          title: "CPU1",
          type: "area",
          data: [
            {
              x: new Date(1600963200000),
              y: 0.11879533073929961,
            },
            {
              x: new Date(1600964100000),
              y: 0.14208560311284046,
            },
            {
              x: new Date(1600965000000),
              y: 0.14633047989623865,
            },
            {
              x: new Date(1600965900000),
              y: 0.12792529182879378,
            },
            {
              x: new Date(1600966800000),
              y: 0.1265431906614786,
            },
            {
              x: new Date(1600967700000),
              y: 0.12437665369649804,
            },
            {
              x: new Date(1600968600000),
              y: 0.13730324254215304,
            },
            {
              x: new Date(1600969500000),
              y: 0.1316513618677043,
            },
            {
              x: new Date(1600970400000),
              y: 0.14375408560311284,
            },
            {
              x: new Date(1600971300000),
              y: 0.14973696498054476,
            },
            {
              x: new Date(1600972200000),
              y: 0.12567367055771725,
            },
            {
              x: new Date(1600973100000),
              y: 0.11769649805447471,
            },
            {
              x: new Date(1600974000000),
              y: 0.1410230869001297,
            },
            {
              x: new Date(1600974900000),
              y: 0.1173810635538262,
            },
            {
              x: new Date(1600975800000),
              y: 0.12403424124513618,
            },
            {
              x: new Date(1600976700000),
              y: 0.12901478599221788,
            },
            {
              x: new Date(1600977600000),
              y: 0.13891984435797666,
            },
            {
              x: new Date(1600978500000),
              y: 0.14056861219195851,
            },
            {
              x: new Date(1600979400000),
              y: 0.1361214007782101,
            },
            {
              x: new Date(1600980300000),
              y: 0.14129805447470817,
            },
            {
              x: new Date(1600981200000),
              y: 0.14985421530479898,
            },
            {
              x: new Date(1600982100000),
              y: 0.11973229571984437,
            },
            {
              x: new Date(1600983000000),
              y: 0.14454682230869,
            },
            {
              x: new Date(1600983900000),
              y: 0.1333509727626459,
            },
            {
              x: new Date(1600984800000),
              y: 0.11119066147859921,
            },
            {
              x: new Date(1600985700000),
              y: 0.11443112840466925,
            },
            {
              x: new Date(1600986600000),
              y: 0.13957250324254214,
            },
            {
              x: new Date(1600987500000),
              y: 0.11549779507133592,
            },
            {
              x: new Date(1600988400000),
              y: 0.14805291828793773,
            },
            {
              x: new Date(1600989300000),
              y: 0.1355662775616083,
            },
            {
              x: new Date(1600990200000),
              y: 0.15503813229571983,
            },
            {
              x: new Date(1600991100000),
              y: 0.12650894941634241,
            },
          ],
          valueFormatter: function percentageFormatter(e) {
            return (100 * e).toFixed(0) + "%";
          },
        },
        {
          title: "CPU2",
          type: "area",
          data: [
            {
              x: new Date(1600963200000),
              y: 0.1313857328145266,
            },
            {
              x: new Date(1600964100000),
              y: 0.15256549935149158,
            },
            {
              x: new Date(1600965000000),
              y: 0.14162178988326848,
            },
            {
              x: new Date(1600965900000),
              y: 0.15541167315175097,
            },
            {
              x: new Date(1600966800000),
              y: 0.13147600518806743,
            },
            {
              x: new Date(1600967700000),
              y: 0.11202490272373541,
            },
            {
              x: new Date(1600968600000),
              y: 0.13693281452658884,
            },
            {
              x: new Date(1600969500000),
              y: 0.15257587548638132,
            },
            {
              x: new Date(1600970400000),
              y: 0.12727470817120623,
            },
            {
              x: new Date(1600971300000),
              y: 0.1166651102464332,
            },
            {
              x: new Date(1600972200000),
              y: 0.13817587548638133,
            },
            {
              x: new Date(1600973100000),
              y: 0.10018988326848248,
            },
            {
              x: new Date(1600974000000),
              y: 0.12684409857328144,
            },
            {
              x: new Date(1600974900000),
              y: 0.1029810635538262,
            },
            {
              x: new Date(1600975800000),
              y: 0.15013540856031127,
            },
            {
              x: new Date(1600976700000),
              y: 0.15517198443579766,
            },
            {
              x: new Date(1600977600000),
              y: 0.1298396887159533,
            },
            {
              x: new Date(1600978500000),
              y: 0.12954085603112842,
            },
            {
              x: new Date(1600979400000),
              y: 0.10663346303501946,
            },
            {
              x: new Date(1600980300000),
              y: 0.09411984435797666,
            },
            {
              x: new Date(1600981200000),
              y: 0.12106977950713359,
            },
            {
              x: new Date(1600982100000),
              y: 0.13847678339818417,
            },
            {
              x: new Date(1600983000000),
              y: 0.11526848249027237,
            },
            {
              x: new Date(1600983900000),
              y: 0.10887782101167316,
            },
            {
              x: new Date(1600984800000),
              y: 0.13845291828793774,
            },
            {
              x: new Date(1600985700000),
              y: 0.14058728923476005,
            },
            {
              x: new Date(1600986600000),
              y: 0.10500440985732813,
            },
            {
              x: new Date(1600987500000),
              y: 0.09681556420233463,
            },
            {
              x: new Date(1600988400000),
              y: 0.10691880674448768,
            },
            {
              x: new Date(1600989300000),
              y: 0.14403112840466925,
            },
            {
              x: new Date(1600990200000),
              y: 0.11747756160830092,
            },
            {
              x: new Date(1600991100000),
              y: 0.14765654993514915,
            },
          ],
          valueFormatter: function percentageFormatter(e) {
            return (100 * e).toFixed(0) + "%";
          },
        },
        {
          title: "CPU3",
          type: "area",
          data: [
            {
              x: new Date(1600963200000),
              y: 0.010804669260700388,
            },
            {
              x: new Date(1600964100000),
              y: 0.02758184176394293,
            },
            {
              x: new Date(1600965000000),
              y: 0.04730791180285344,
            },
            {
              x: new Date(1600965900000),
              y: 0.06839740596627757,
            },
            {
              x: new Date(1600966800000),
              y: 0.07909001297016861,
            },
            {
              x: new Date(1600967700000),
              y: 0.06473151750972762,
            },
            {
              x: new Date(1600968600000),
              y: 0.08646433203631647,
            },
            {
              x: new Date(1600969500000),
              y: 0.13199377431906614,
            },
            {
              x: new Date(1600970400000),
              y: 0.10874396887159532,
            },
            {
              x: new Date(1600971300000),
              y: 0.15138677042801554,
            },
            {
              x: new Date(1600972200000),
              y: 0.1259403372243839,
            },
            {
              x: new Date(1600973100000),
              y: 0.1172171206225681,
            },
            {
              x: new Date(1600974000000),
              y: 0.15072684824902724,
            },
            {
              x: new Date(1600974900000),
              y: 0.14481141374837875,
            },
            {
              x: new Date(1600975800000),
              y: 0.1331704280155642,
            },
            {
              x: new Date(1600976700000),
              y: 0.12739195849546045,
            },
            {
              x: new Date(1600977600000),
              y: 0.15086485084306095,
            },
            {
              x: new Date(1600978500000),
              y: 0.18314811932555122,
            },
            {
              x: new Date(1600979400000),
              y: 0.20856653696498054,
            },
            {
              x: new Date(1600980300000),
              y: 0.20393047989623866,
            },
            {
              x: new Date(1600981200000),
              y: 0.22181374837872891,
            },
            {
              x: new Date(1600982100000),
              y: 0.21278962386511024,
            },
            {
              x: new Date(1600983000000),
              y: 0.22450739299610895,
            },
            {
              x: new Date(1600983900000),
              y: 0.1653810635538262,
            },
            {
              x: new Date(1600984800000),
              y: 0.2478360570687419,
            },
            {
              x: new Date(1600985700000),
              y: 0.21530479896238652,
            },
            {
              x: new Date(1600986600000),
              y: 0.19414785992217898,
            },
            {
              x: new Date(1600987500000),
              y: 0.3259818417639429,
            },
            {
              x: new Date(1600988400000),
              y: 0.17188378728923476,
            },
            {
              x: new Date(1600989300000),
              y: 0.18218832684824904,
            },
            {
              x: new Date(1600990200000),
              y: 0.238694682230869,
            },
            {
              x: new Date(1600991100000),
              y: 0.3049328145265888,
            },
          ],
          valueFormatter: function percentageFormatter(e) {
            return (100 * e).toFixed(0) + "%";
          },
        },
        {
          title: "CPU4",
          type: "area",
          data: [
            {
              x: new Date(1600963200000),
              y: 0.4169463035019455,
            },
            {
              x: new Date(1600964100000),
              y: 0.2213769130998703,
            },
            {
              x: new Date(1600965000000),
              y: 0.16022723735408562,
            },
            {
              x: new Date(1600965900000),
              y: 0.11987963683527886,
            },
            {
              x: new Date(1600966800000),
              y: 0.14835382619974058,
            },
            {
              x: new Date(1600967700000),
              y: 0.10211258106355382,
            },
            {
              x: new Date(1600968600000),
              y: 0.07866044098573281,
            },
            {
              x: new Date(1600969500000),
              y: 0.06610428015564201,
            },
            {
              x: new Date(1600970400000),
              y: 0.07220752269779507,
            },
            {
              x: new Date(1600971300000),
              y: 0.07288715953307393,
            },
            {
              x: new Date(1600972200000),
              y: 0.04931984435797666,
            },
            {
              x: new Date(1600973100000),
              y: 0.05281867704280156,
            },
            {
              x: new Date(1600974000000),
              y: 0.04482075226977951,
            },
            {
              x: new Date(1600974900000),
              y: 0.04784954604409857,
            },
            {
              x: new Date(1600975800000),
              y: 0.043612970168612195,
            },
            {
              x: new Date(1600976700000),
              y: 0.03713203631647211,
            },
            {
              x: new Date(1600977600000),
              y: 0.04359429312581064,
            },
            {
              x: new Date(1600978500000),
              y: 0.039395071335927366,
            },
            {
              x: new Date(1600979400000),
              y: 0.037094682230869,
            },
            {
              x: new Date(1600980300000),
              y: 0.03044980544747082,
            },
            {
              x: new Date(1600981200000),
              y: 0.02587081712062257,
            },
            {
              x: new Date(1600982100000),
              y: 0.028407782101167314,
            },
            {
              x: new Date(1600983000000),
              y: 0.025798184176394293,
            },
            {
              x: new Date(1600983900000),
              y: 0.02083942931258106,
            },
            {
              x: new Date(1600984800000),
              y: 0.02171413748378729,
            },
            {
              x: new Date(1600985700000),
              y: 0.020902723735408562,
            },
            {
              x: new Date(1600986600000),
              y: 0.025170428015564204,
            },
            {
              x: new Date(1600987500000),
              y: 0.022062775616083007,
            },
            {
              x: new Date(1600988400000),
              y: 0.020867444876783398,
            },
            {
              x: new Date(1600989300000),
              y: 0.022508949416342416,
            },
            {
              x: new Date(1600990200000),
              y: 0.019239429312581064,
            },
            {
              x: new Date(1600991100000),
              y: 0.018333592736705578,
            },
          ],
          valueFormatter: function percentageFormatter(e) {
            return (100 * e).toFixed(0) + "%";
          },
        },
        {
          title: "CPU5",
          type: "area",
          data: [
            {
              x: new Date(1600963200000),
              y: 0.035286121919584953,
            },
            {
              x: new Date(1600964100000),
              y: 0.029357198443579768,
            },
            {
              x: new Date(1600965000000),
              y: 0.03726070038910506,
            },
            {
              x: new Date(1600965900000),
              y: 0.048325810635538265,
            },
            {
              x: new Date(1600966800000),
              y: 0.03768300907911803,
            },
            {
              x: new Date(1600967700000),
              y: 0.03875175097276264,
            },
            {
              x: new Date(1600968600000),
              y: 0.04819818417639429,
            },
            {
              x: new Date(1600969500000),
              y: 0.07545421530479897,
            },
            {
              x: new Date(1600970400000),
              y: 0.060908949416342416,
            },
            {
              x: new Date(1600971300000),
              y: 0.09228845654993516,
            },
            {
              x: new Date(1600972200000),
              y: 0.1118651102464332,
            },
            {
              x: new Date(1600973100000),
              y: 0.12358495460440987,
            },
            {
              x: new Date(1600974000000),
              y: 0.1346241245136187,
            },
            {
              x: new Date(1600974900000),
              y: 0.1371569390402075,
            },
            {
              x: new Date(1600975800000),
              y: 0.21250739299610896,
            },
            {
              x: new Date(1600976700000),
              y: 0.37775771725032425,
            },
            {
              x: new Date(1600977600000),
              y: 0.3320861219195849,
            },
            {
              x: new Date(1600978500000),
              y: 0.39978002594033724,
            },
            {
              x: new Date(1600979400000),
              y: 0.14514552529182878,
            },
            {
              x: new Date(1600980300000),
              y: 0.1176207522697795,
            },
            {
              x: new Date(1600981200000),
              y: 0.09372140077821012,
            },
            {
              x: new Date(1600982100000),
              y: 0.08439844357976653,
            },
            {
              x: new Date(1600983000000),
              y: 0.09929961089494163,
            },
            {
              x: new Date(1600983900000),
              y: 0.0669260700389105,
            },
            {
              x: new Date(1600984800000),
              y: 0.06256186770428016,
            },
            {
              x: new Date(1600985700000),
              y: 0.060151491569390404,
            },
            {
              x: new Date(1600986600000),
              y: 0.06549105058365759,
            },
            {
              x: new Date(1600987500000),
              y: 0.04230350194552529,
            },
            {
              x: new Date(1600988400000),
              y: 0.04555849546044098,
            },
            {
              x: new Date(1600989300000),
              y: 0.030188326848249028,
            },
            {
              x: new Date(1600990200000),
              y: 0.034107392996108946,
            },
            {
              x: new Date(1600991100000),
              y: 0.044323735408560314,
            },
          ],
          valueFormatter: function percentageFormatter(e) {
            return (100 * e).toFixed(0) + "%";
          },
        },
        {
          title: "CPU6",
          type: "area",
          data: [
            {
              x: new Date(1600963200000),
              y: 0.10952944228274968,
            },
            {
              x: new Date(1600964100000),
              y: 0.1410614785992218,
            },
            {
              x: new Date(1600965000000),
              y: 0.10770635538261997,
            },
            {
              x: new Date(1600965900000),
              y: 0.1404171206225681,
            },
            {
              x: new Date(1600966800000),
              y: 0.11001504539559015,
            },
            {
              x: new Date(1600967700000),
              y: 0.11049546044098574,
            },
            {
              x: new Date(1600968600000),
              y: 0.14370946822308692,
            },
            {
              x: new Date(1600969500000),
              y: 0.14083320363164722,
            },
            {
              x: new Date(1600970400000),
              y: 0.11042697795071335,
            },
            {
              x: new Date(1600971300000),
              y: 0.14704643320363164,
            },
            {
              x: new Date(1600972200000),
              y: 0.1471118028534371,
            },
            {
              x: new Date(1600973100000),
              y: 0.14830817120622566,
            },
            {
              x: new Date(1600974000000),
              y: 0.10513618677042802,
            },
            {
              x: new Date(1600974900000),
              y: 0.1139911802853437,
            },
            {
              x: new Date(1600975800000),
              y: 0.15371206225680933,
            },
            {
              x: new Date(1600976700000),
              y: 0.15401089494163422,
            },
            {
              x: new Date(1600977600000),
              y: 0.10528664072632944,
            },
            {
              x: new Date(1600978500000),
              y: 0.10756731517509728,
            },
            {
              x: new Date(1600979400000),
              y: 0.10576601815823607,
            },
            {
              x: new Date(1600980300000),
              y: 0.1062028534370947,
            },
            {
              x: new Date(1600981200000),
              y: 0.1485011673151751,
            },
            {
              x: new Date(1600982100000),
              y: 0.1044669260700389,
            },
            {
              x: new Date(1600983000000),
              y: 0.11177587548638133,
            },
            {
              x: new Date(1600983900000),
              y: 0.10890376134889754,
            },
            {
              x: new Date(1600984800000),
              y: 0.14598495460440986,
            },
            {
              x: new Date(1600985700000),
              y: 0.10602749675745785,
            },
            {
              x: new Date(1600986600000),
              y: 0.1505431906614786,
            },
            {
              x: new Date(1600987500000),
              y: 0.11074656290531777,
            },
            {
              x: new Date(1600988400000),
              y: 0.11062827496757459,
            },
            {
              x: new Date(1600989300000),
              y: 0.106021271076524,
            },
            {
              x: new Date(1600990200000),
              y: 0.14992892347600517,
            },
            {
              x: new Date(1600991100000),
              y: 0.14492347600518807,
            },
          ],
          valueFormatter: function percentageFormatter(e) {
            return (100 * e).toFixed(0) + "%";
          },
        },
      ]}
      xDomain={[new Date(1600963200000), new Date(1600991100000)]}
      yDomain={[0, 1]}
      i18nStrings={{
        xTickFormatter: (e) =>
          e
            .toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
              hour: "numeric",
              minute: "numeric",
              hour12: !1,
            })
            .split(",")
            .join("\n"),
        yTickFormatter: function percentageFormatter(e) {
          return (100 * e).toFixed(0) + "%";
        },
      }}
      ariaLabel="Stacked area chart, multiple metrics"
      height={300}
      xScaleType="time"
      xTitle="Time (UTC)"
      yTitle="Total CPU load"
      empty={
        <Box textAlign="center" color="inherit">
          <b>No data available</b>
          <Box variant="p" color="inherit">
            There is no data available
          </Box>
        </Box>
      }
      noMatch={
        <Box textAlign="center" color="inherit">
          <b>No matching data</b>
          <Box variant="p" color="inherit">
            There is no matching data to display
          </Box>
          <Button>Clear filter</Button>
        </Box>
      }
    />
  );
};
export default ChartPage;
