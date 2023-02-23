# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from mab2rec import BanditRecommender, LearningPolicy, NeighborhoodPolicy
from tests.test_base import BaseTest


class BanditRecommenderTest(BaseTest):

    def test_init(self):
        rec = BanditRecommender(LearningPolicy.UCB1())
        rec._init([1, 2, 3])
        self.assertTrue(rec.mab is not None)

    def test_learning_policies(self):
        for lp in self.lps:
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                         rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         top_k=2,
                         seed=123456)

    def test_parametric_learning_policies(self):
        for lp in self.para_lps:
            self.predict(arms=[1, 2, 3],
                         decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                         rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                       [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                       [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                       [0, 2, 1, 0, 0]],
                         top_k=2,
                         seed=123456)

    def test_neighborhood_policies(self):
        for cp in self.nps:
            for lp in self.lps:
                if not self.is_compatible(lp, cp):
                    continue
                self.predict(arms=[1, 2, 3],
                             decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                             rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                           [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                           [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                           [0, 2, 1, 0, 0]],
                             top_k=2,
                             seed=123456)
            for lp in self.para_lps:
                if not self.is_compatible(lp, cp):
                    continue
                self.predict(arms=[1, 2, 3],
                             decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                             rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                           [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                           [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                           [0, 2, 1, 0, 0]],
                             top_k=2,
                             seed=123456)

    def test_learning_policies_predict(self):
        for lp in self.lps:
            _, rec = self.predict(arms=[1, 2, 3],
                                  decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                  rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                  learning_policy=lp,
                                  top_k=2,
                                  seed=123456)
            rec.predict()
            rec.predict_expectations()

    def test_parametric_learning_policies_predict(self):
        for lp in self.para_lps:
            _, rec = self.predict(arms=[1, 2, 3],
                                  decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                  rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                  learning_policy=lp,
                                  contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                  context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                [0, 2, 1, 0, 0]],
                                  top_k=2,
                                  seed=123456)
            rec.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [1, 2, 1, 1, 3]])
            rec.predict_expectations([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [1, 2, 1, 1, 3]])

    def test_neighborhood_policies_predict(self):
        for cp in self.nps:
            for lp in self.lps:
                if not self.is_compatible(lp, cp):
                    continue
                _, rec = self.predict(arms=[1, 2, 3],
                                      decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                      rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                      learning_policy=lp,
                                      neighborhood_policy=cp,
                                      contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                      context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                    [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                    [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                    [0, 2, 1, 0, 0]],
                                      top_k=2,
                                      seed=123456)
                rec.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [1, 2, 1, 1, 3]])
                rec.predict_expectations([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [1, 2, 1, 1, 3]])
            for lp in self.para_lps:
                if not self.is_compatible(lp, cp):
                    continue
                _, rec = self.predict(arms=[1, 2, 3],
                                      decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                      rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                      learning_policy=lp,
                                      neighborhood_policy=cp,
                                      contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                      context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                    [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                    [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                    [0, 2, 1, 0, 0]],
                                      top_k=2,
                                      seed=123456)
                rec.predict([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [1, 2, 1, 1, 3]])
                rec.predict_expectations([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [1, 2, 1, 1, 3]])

    def test_learning_policies_partial_fit(self):
        for lp in self.lps:
            _, rec = self.predict(arms=[1, 2, 3],
                                  decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                  rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                  learning_policy=lp,
                                  top_k=2,
                                  seed=123456)
            rec.partial_fit(decisions=[1, 1, 2], rewards=[0, 1, 0])

    def test_parametric_learning_policies_partial_fit(self):
        for lp in self.para_lps:
            _, rec = self.predict(arms=[1, 2, 3],
                                  decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                  rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                  learning_policy=lp,
                                  contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                  context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                [0, 2, 1, 0, 0]],
                                  top_k=2,
                                  seed=123456)
            rec.partial_fit(decisions=[1, 1, 2], rewards=[0, 1, 0],
                            contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [1, 2, 1, 1, 3]])

    def test_neighborhood_policies_partial_fit(self):
        for cp in self.nps:
            for lp in self.lps:
                if not self.is_compatible(lp, cp):
                    continue
                _, rec = self.predict(arms=[1, 2, 3],
                                      decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                      rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                      learning_policy=lp,
                                      neighborhood_policy=cp,
                                      contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                      context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                    [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                    [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                    [0, 2, 1, 0, 0]],
                                      top_k=2,
                                      seed=123456)
                rec.partial_fit(decisions=[1, 1, 2], rewards=[0, 1, 0],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [1, 2, 1, 1, 3]])
            for lp in self.para_lps:
                if not self.is_compatible(lp, cp):
                    continue
                _, rec = self.predict(arms=[1, 2, 3],
                                      decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                      rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                      learning_policy=lp,
                                      neighborhood_policy=cp,
                                      contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                      context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                    [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                    [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                    [0, 2, 1, 0, 0]],
                                      top_k=2,
                                      seed=123456)
                rec.partial_fit(decisions=[1, 1, 2], rewards=[0, 1, 0],
                                contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [1, 2, 1, 1, 3]])

    def test_learning_policies_warm_start(self):
        for lp in self.lps:
            self.predict(arms=[1, 2, 3, 4],
                         decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                         rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         top_k=2,
                         seed=123456,
                         warm_start=True,
                         arm_to_features={1: [0.5, 1], 2: [0, 1], 3: [0, 1], 4: [0.5, 1]})

    def test_parametric_learning_policies_warm_start(self):
        for lp in self.para_lps:
            self.predict(arms=[1, 2, 3, 4],
                         decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                         rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                         learning_policy=lp,
                         contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                         context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                       [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                       [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                       [0, 2, 1, 0, 0]],
                         top_k=2,
                         seed=123456,
                         warm_start=True,
                         arm_to_features={1: [0.5, 1], 2: [0, 1], 3: [0, 1], 4: [0.5, 1]})

    def test_neighborhood_policies_warm_start(self):
        for cp in self.nps:
            for lp in self.lps:
                if not self.is_compatible(lp, cp):
                    continue
                self.predict(arms=[1, 2, 3, 4],
                             decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                             rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                           [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                           [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                           [0, 2, 1, 0, 0]],
                             top_k=2,
                             seed=123456,
                             warm_start=True,
                             arm_to_features={1: [0.5, 1], 2: [0, 1], 3: [0, 1], 4: [0.5, 1]})
            for lp in self.para_lps:
                if not self.is_compatible(lp, cp):
                    continue
                self.predict(arms=[1, 2, 3, 4],
                             decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                             rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                             learning_policy=lp,
                             neighborhood_policy=cp,
                             contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                             context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                           [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                           [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                           [0, 2, 1, 0, 0]],
                             top_k=2,
                             seed=123456,
                             warm_start=True,
                             arm_to_features={1: [0.5, 1], 2: [0, 1], 3: [0, 1], 4: [0.5, 1]})

    def test_learning_policies_add_arm(self):
        for lp in self.lps:
            rec = BanditRecommender(lp)
            rec._init(arms=[1, 2, 3])
            rec.add_arm(4)
            self.assertEqual(len(rec.mab.arms), 4)
            self.assertTrue(4 in rec.mab.arms)
            self.assertTrue(4 in rec.mab._imp.arm_to_expectation)
        for lp in self.lps:
            rec = BanditRecommender(lp)
            rec.add_arm(4)
            self.assertEqual(len(rec.mab.arms), 1)
            self.assertTrue(4 in rec.mab.arms)
            self.assertTrue(4 in rec.mab._imp.arm_to_expectation)

    def test_parametric_learning_policies_add_arm(self):
        for lp in self.para_lps:
            rec = BanditRecommender(lp)
            rec._init(arms=[1, 2, 3])
            rec.add_arm(4)
            self.assertEqual(len(rec.mab.arms), 4)
            self.assertTrue(4 in rec.mab.arms)
            self.assertTrue(4 in rec.mab._imp.arm_to_expectation)
            self.assertTrue(4 in rec.mab._imp.arm_to_model)
        for lp in self.para_lps:
            rec = BanditRecommender(lp)
            rec.add_arm(4)
            self.assertEqual(len(rec.mab.arms), 1)
            self.assertTrue(4 in rec.mab.arms)
            self.assertTrue(4 in rec.mab._imp.arm_to_expectation)
            self.assertTrue(4 in rec.mab._imp.arm_to_model)

    def test_neighborhood_policies_add_arm(self):
        for cp in self.nps:
            for lp in self.lps:
                if not self.is_compatible(lp, cp):
                    continue
                rec = BanditRecommender(lp)
                rec._init(arms=[1, 2, 3])
                rec.add_arm(4)
                self.assertEqual(len(rec.mab.arms), 4)
                self.assertTrue(4 in rec.mab.arms)
                self.assertTrue(4 in rec.mab._imp.arm_to_expectation)
        for cp in self.nps:
            for lp in self.lps:
                if not self.is_compatible(lp, cp):
                    continue
                rec = BanditRecommender(lp)
                rec.add_arm(4)
                self.assertEqual(len(rec.mab.arms), 1)
                self.assertTrue(4 in rec.mab.arms)
                self.assertTrue(4 in rec.mab._imp.arm_to_expectation)

    def test_learning_policies_remove_arm(self):
        for lp in self.lps:
            rec = BanditRecommender(lp)
            rec._init(arms=[1, 2, 3])
            rec.remove_arm(3)
            self.assertTrue(3 not in rec.mab.arms)
            self.assertTrue(3 not in rec.mab._imp.arm_to_expectation)

    def test_parametric_learning_policies_remove_arm(self):
        for lp in self.para_lps:
            rec = BanditRecommender(lp)
            rec._init(arms=[1, 2, 3])
            rec.remove_arm(3)
            self.assertTrue(3 not in rec.mab.arms)
            self.assertTrue(3 not in rec.mab._imp.arm_to_expectation)
            self.assertTrue(3 not in rec.mab._imp.arm_to_model)

    def test_neighborhood_policies_remove_arm(self):
        for cp in self.nps:
            for lp in self.lps:
                if not self.is_compatible(lp, cp):
                    continue
                rec = BanditRecommender(lp)
                rec._init(arms=[1, 2, 3])
                rec.remove_arm(3)
                self.assertTrue(3 not in rec.mab.arms)
                self.assertTrue(3 not in rec.mab._imp.arm_to_expectation)

    def test_learning_policies_set_arms(self):
        for lp in self.lps:
            _, rec = self.predict(arms=[1, 2, 3, 4],
                                  decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                  rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                  learning_policy=lp,
                                  contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                  top_k=2,
                                  seed=123456)
            rec.set_arms([2, 5])
            self.assertEqual(rec.mab.arms, [2, 5])
            self.assertEqual(rec.mab._imp.arms, [2, 5])

    def test_parametric_learning_policies_set_arms(self):
        for lp in self.para_lps:
            _, rec = self.predict(arms=[1, 2, 3],
                                  decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                  rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                  learning_policy=lp,
                                  contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                  context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                [0, 2, 1, 0, 0]],
                                  top_k=2,
                                  seed=123456)
            rec.set_arms([2, 5])
            self.assertEqual(rec.mab.arms, [2, 5])
            self.assertEqual(rec.mab._imp.arms, [2, 5])

    def test_neighborhood_policies_set_arms(self):
        for cp in self.nps:
            for lp in self.lps:
                if not self.is_compatible(lp, cp):
                    continue
                _, rec = self.predict(arms=[1, 2, 3],
                                      decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                      rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                      learning_policy=lp,
                                      neighborhood_policy=cp,
                                      contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                      context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                       [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                       [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                       [0, 2, 1, 0, 0]],
                                      top_k=2,
                                      seed=123456)
                rec.set_arms([2, 5])
                self.assertEqual(rec.mab.arms, [2, 5])
                self.assertEqual(rec.mab._imp.arms, [2, 5])
            for lp in self.para_lps:
                if not self.is_compatible(lp, cp):
                    continue
                _, rec = self.predict(arms=[1, 2, 3],
                                      decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                      rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                      learning_policy=lp,
                                      neighborhood_policy=cp,
                                      contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                      context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                       [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                       [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                       [0, 2, 1, 0, 0]],
                                      top_k=2,
                                      seed=123456)
                rec.set_arms([2, 5])
                self.assertEqual(rec.mab.arms, [2, 5])
                self.assertEqual(rec.mab._imp.arms, [2, 5])

    def test_recommend_random(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.Random(),
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [1, 2])
        self.assertListAlmostEqual(results[1], [0.6539649643958056, 0.5950330918093963])

        # No scores
        results = rec.recommend(return_scores=False)
        self.assertEqual(results, [1, 2])

    def test_recommend_random_w_empty_context(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.Random(),
                                    contexts=[[]] * 5,
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [[1, 2], [1, 2], [3, 2], [1, 3], [2, 3]])

    def test_recommend_popularity(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.Popularity(),
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [3, 1])
        self.assertListAlmostEqual(results[1], [0.6871905288086486, 0.5530453076346826])

        # No scores
        results = rec.recommend(return_scores=False)
        self.assertEqual(results, [1, 3])

    def test_recommend_popularity_w_empty_context(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.Popularity(),
                                    contexts=[[]] * 5,
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [[3, 1], [1, 3], [3, 1], [3, 1], [3, 1]])

    def test_recommend_greedy(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0),
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [1, 3])
        self.assertListAlmostEqual(results[1], [0.6607563687658172, 0.6456563062257954])

        # No scores
        results = rec.recommend(return_scores=False)
        self.assertEqual(results, [1, 3])

    def test_recommend_greedy_eps(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [1, 3])
        self.assertListAlmostEqual(results[1], [0.6607563687658172, 0.6456563062257954])

        # No scores
        results = rec.recommend(return_scores=False)
        self.assertEqual(results, [2, 3])

    def test_recommend_greedy_w_empty_context(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.5),
                                    contexts=[[]] * 5,
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [[1, 3], [1, 2], [3, 1], [1, 3], [1, 3]])

    def test_recommend_softmax(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.Softmax(),
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [3, 1])
        self.assertListAlmostEqual(results[1], [0.6927815920738961, 0.5465755461646858])

        # No scores
        results = rec.recommend(return_scores=False)
        self.assertEqual(results, [2, 3])

    def test_recommend_softmax_w_empty_context(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.Softmax(),
                                    contexts=[[]] * 5,
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [[3, 1], [2, 3], [3, 2], [3, 2], [3, 1]])

    def test_recommend_ucb(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.UCB1(),
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [1, 3])
        self.assertListAlmostEqual(results[1], [0.8705286139301878, 0.8263110445099603])

        # No scores
        results = rec.recommend(return_scores=False)
        self.assertEqual(results, [1, 3])

    def test_recommend_ucb_alpha(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.UCB1(alpha=10),
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [2, 1])
        self.assertListAlmostEqual(results[1], [0.9999997430210213, 0.9999978636449579])

        # No scores
        results = rec.recommend(return_scores=False)
        self.assertEqual(results, [2, 1])

    def test_recommend_ucb_w_empty_context(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.UCB1(),
                                    contexts=[[]] * 5,
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [[1, 3], [1, 3], [1, 3], [1, 3], [1, 3]])

    def test_recommend_ts(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.ThompsonSampling(),
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [1, 2])
        self.assertListAlmostEqual(results[1], [0.6378726378265008, 0.5875821922391576])

        # No scores
        results = rec.recommend(return_scores=False)
        self.assertEqual(results, [3, 1])

    def test_recommend_ts_w_empty_context(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.ThompsonSampling(),
                                    contexts=[[]] * 5,
                                    neighborhood_policy=None,
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [[1, 3], [3, 1], [1, 3], [3, 2], [1, 2]])

    def test_recommend_lin_greedy(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.LinGreedy(epsilon=0.5),
                                    neighborhood_policy=None,
                                    contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                    context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                     [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                     [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                     [0, 2, 1, 0, 0]],
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [[3, 1], [2, 3]])
        self.assertListAlmostEqual(results[1][0], [0.6504125435586658, 0.5240639631785098])
        self.assertListAlmostEqual(results[1][1], [0.7221703184615302, 0.7121913829791641])

        # No scores
        results = rec.recommend([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]], return_scores=False)
        self.assertEqual(results, [[2, 3], [2, 1]])

    def test_recommend_lin_ucb(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.LinUCB(alpha=1.25),
                                    neighborhood_policy=None,
                                    contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                    context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                     [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                     [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                     [0, 2, 1, 0, 0]],
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [[3, 2], [1, 3]])
        self.assertListAlmostEqual(results[1][0], [0.8355754378823774, 0.8103388262282213])
        self.assertListAlmostEqual(results[1][1], [0.8510415343853225, 0.8454457789037026])

        # No scores
        results = rec.recommend([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]], return_scores=False)
        self.assertEqual(results, [[3, 2], [1, 3]])

    def test_recommend_lin_ts(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.LinTS(),
                                    neighborhood_policy=None,
                                    contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                    context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                     [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                     [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                     [0, 2, 1, 0, 0]],
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [[3, 2], [1, 3]])
        self.assertListAlmostEqual(results[1][0], [0.6393327956234724, 0.6113795857188596])
        self.assertListAlmostEqual(results[1][1], [0.9231640190086698, 0.9093145340785204])

        # No scores
        results = rec.recommend([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]], return_scores=False)
        self.assertEqual(results, [[3, 1], [1, 3]])

    def test_recommend_clusters_ts(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.ThompsonSampling(),
                                    neighborhood_policy=NeighborhoodPolicy.Clusters(),
                                    contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                    context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                     [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                     [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                     [0, 2, 1, 0, 0]],
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [[2, 1], [1, 3]])
        self.assertListAlmostEqual(results[1][0], [0.6470729583134509, 0.6239486262002204])
        self.assertListAlmostEqual(results[1][1], [0.7257397617770284, 0.6902019029795886])

        # No scores
        results = rec.recommend([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]], return_scores=False)
        self.assertEqual(results, [[2, 3], [1, 3]])

    def test_recommend_radius_ts(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.ThompsonSampling(),
                                    neighborhood_policy=NeighborhoodPolicy.Radius(radius=5),
                                    contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                    context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                     [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                     [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                     [0, 2, 1, 0, 0]],
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [[1, 3], [3, 2]])
        self.assertListAlmostEqual(results[1][0], [0.6853064650518793, 0.5794087793326232])
        self.assertListAlmostEqual(results[1][1], [0.6171485591737581, 0.6039485772665535])

        # No scores
        results = rec.recommend([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]], return_scores=False)
        self.assertEqual(results, [[3, 1], [2, 1]])

    def test_recommend_knn_ts(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.ThompsonSampling(),
                                    neighborhood_policy=NeighborhoodPolicy.KNearest(k=2),
                                    contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                    context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                     [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                     [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                     [0, 2, 1, 0, 0]],
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [[2, 1], [1, 3]])
        self.assertListAlmostEqual(results[1][0], [0.6470729583134509, 0.6239486262002204])
        self.assertListAlmostEqual(results[1][1], [0.7257397617770284, 0.7239071840518659])

        # No scores
        results = rec.recommend([[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]], return_scores=False)
        self.assertEqual(results, [[3, 2], [1, 3]])

    def test_recommend_lin_ucb_excluded(self):
        results, rec = self.predict(arms=[1, 2, 3],
                                    decisions=[1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                                    rewards=[0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                    learning_policy=LearningPolicy.LinUCB(alpha=1.25),
                                    neighborhood_policy=None,
                                    contexts=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1]],
                                    context_history=[[0, 1, 2, 3, 5], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                                                     [0, 2, 2, 3, 5], [1, 3, 1, 1, 1], [0, 0, 0, 0, 0],
                                                     [0, 1, 4, 3, 5], [0, 1, 2, 4, 5], [1, 2, 1, 1, 3],
                                                     [0, 2, 1, 0, 0]],
                                    excluded_arms=[[3], [3]],
                                    top_k=2,
                                    seed=123456)
        self.assertEqual(results[0], [[2, 1], [1, 2]])
        self.assertListAlmostEqual(results[1][0], [0.8103388262282213, 0.7882460646490305])

    def test_recommend_exclusion_replace(self):
        results, rec = self.predict(arms=[1, 2, 3, 4, 5],
                                    decisions=[1, 2, 3, 4, 5],
                                    rewards=[0.9, 0.8, 0.7, 0.6, 0.5],
                                    learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.),
                                    neighborhood_policy=None,
                                    contexts=[[1, 2], [1, 2]],
                                    excluded_arms=[[], [1]],
                                    top_k=4)

        self.assertListEqual(results[0][0], [1, 2, 3, 4])
        self.assertListEqual(results[0][1], [2, 3, 4, 5])
