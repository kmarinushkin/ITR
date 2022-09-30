import warnings
import os
import unittest
import pandas as pd
import numpy as np

from ITR.interfaces import ETimeFrames, EScope
from ITR.temperature_score import TemperatureScore
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.data.osc_units import ureg, Q_
from utils import assert_pint_frame_equal


class TestTemperatureScore(unittest.TestCase):
    """
    Test the reporting functionality. We'll use the Example data provider as the output of this provider is known in
    advance.
    """

    def setUp(self) -> None:
        """
        Create the provider and reporting instance which we'll use later on.
        :return:
        """
        self.temperature_score = TemperatureScore(time_frames=[ETimeFrames.LONG], scopes=EScope.get_result_scopes())
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "inputs",
                                             "data_test_temperature_score.csv"), sep=";")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['ghg_s1s2'] = df['ghg_s1s2'].astype('pint[t CO2]')
            df['ghg_s3'] = df['ghg_s3'].astype('pint[t CO2]')
            for cumulative in ['cumulative_budget', 'cumulative_target', 'cumulative_trajectory']:
                df[cumulative] = df[cumulative].astype('pint[Mt CO2]')
            df['benchmark_global_budget'] = df['benchmark_global_budget'].astype('pint[Gt CO2]')
            df['benchmark_temperature'] = df['benchmark_temperature'].astype('pint[delta_degC]')
        self.data = df

    def test_temp_score(self) -> None:
        """
        Test whether the temperature score is calculated as expected.

        :return:
        """
        scores = self.temperature_score.calculate(self.data)
        self.assertAlmostEqual(scores[
                                   (scores["company_name"] == "Company T") &
                                   (scores["scope"] == EScope.S1S2)
                                   ]["temperature_score"].iloc[0], Q_(1.82, ureg.delta_degC), places=2, msg="The temp score was incorrect")
        self.assertAlmostEqual(scores[
                                   (scores["company_name"] == "Company E") &
                                   (scores["scope"] == EScope.S1S2)
                                   ]["temperature_score"].iloc[0], Q_(1.84, ureg.delta_degC), places=2,
                               msg="The fallback temp score was incorrect")
        self.assertAlmostEqual(scores[
                                   (scores["company_name"] == "Company AA") &
                                   (scores["time_frame"] == ETimeFrames.LONG) &
                                   (scores["scope"] == EScope.S1S2S3)
                                   ]["temperature_score"].iloc[0], Q_(1.79, ureg.delta_degC), places=5,
                               msg="The aggregated fallback temp score was incorrect")

    def test_temp_score_overwrite_tcre(self) -> None:
        """
        Test whether the temperature score is calculated as  when overwriting the Transient Climate Response cumulative Emissions (TCRE) control.

        :return:
        """
        overwritten_temp_score = self.temperature_score
        overwritten_temp_score.c.CONTROLS_CONFIG.tcre = Q_(1.0, ureg.delta_degC)
        scores = overwritten_temp_score.calculate(self.data)
        self.assertAlmostEqual(scores[
                                   (scores["company_name"] == "Company T") &
                                   (scores["scope"] == EScope.S1S2)
                                   ]["temperature_score"].iloc[0], Q_(1.65, ureg.delta_degC), places=2, msg="The temp score was incorrect")
        self.assertAlmostEqual(scores[
                                   (scores["company_name"] == "Company E") &
                                   (scores["scope"] == EScope.S1S2)
                                   ]["temperature_score"].iloc[0], Q_(1.65, ureg.delta_degC), places=2,
                               msg="The fallback temp score was incorrect")
        self.assertAlmostEqual(scores[
                                   (scores["company_name"] == "Company AA") &
                                   (scores["time_frame"] == ETimeFrames.LONG) &
                                   (scores["scope"] == EScope.S1S2S3)
                                   ]["temperature_score"].iloc[0], Q_(1.63, ureg.delta_degC), places=5,
                               msg="The aggregated fallback temp score was incorrect")

    def test_portfolio_aggregations(self):
        scores = self.temperature_score.calculate(self.data)
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(aggregations.long.S1S2.all.score, Q_(1.857, ureg.delta_degC), places=2,
                               msg="Long WATS aggregation failed")
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.TETS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(aggregations.long.S1S2.all.score, Q_(1.875, ureg.delta_degC), places=2,
                               msg="Long TETS aggregation failed")
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.MOTS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(aggregations.long.S1S2.all.score, Q_(1.869, ureg.delta_degC), places=2,
                               msg="Long MOTS aggregation failed")
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.EOTS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(aggregations.long.S1S2.all.score, Q_(1.840, ureg.delta_degC), places=2,
                               msg="Long EOTS aggregation failed")
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.ECOTS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(aggregations.long.S1S2.all.score, Q_(1.840, ureg.delta_degC), places=2,
                               msg="Long ECOTS aggregation failed")
        self.temperature_score.aggregation_method = PortfolioAggregationMethod.AOTS
        aggregations = self.temperature_score.aggregate_scores(scores)
        self.assertAlmostEqual(aggregations.long.S1S2.all.score, Q_(1.869, ureg.delta_degC), places=2,
                               msg="Long AOTS aggregation failed")

    def test_filter_data(self):
        data = pd.DataFrame(np.array([['id0', ETimeFrames.LONG, EScope.S3, 1],
                                      ['id1', ETimeFrames.MID, EScope.S1S2, 2],
                                      ['id2', ETimeFrames.MID, EScope.S3, 3], # this should stay
                                      ['id3', ETimeFrames.MID, EScope.S3, None]]),
                            index=['id0', 'id1', 'id2', 'id3'], columns=['company_id', 'time_frame', 'scope', 'ghg_s3'])
        expected = pd.DataFrame(np.array([['id2', ETimeFrames.MID, EScope.S3, 3]]),
                                index=['id2'], columns=['company_id', 'time_frame', 'scope', 'ghg_s3'])
        timeframe = ETimeFrames.MID
        scope = EScope.S3

        filtered = self.temperature_score._filter_data(data, timeframe, scope)
        assert_pint_frame_equal(self, filtered, expected)


if __name__ == "__main__":
    test = TestTemperatureScore()
    test.setUp()
    test.test_temp_score()
    test.test_portfolio_aggregations()
