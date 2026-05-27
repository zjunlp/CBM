from swift.template import register_template
from swift.template.templates.qwen import Qwen3_5Template, QwenTemplateMeta


class Qwen3_5KeepHistoryThinkTemplate(Qwen3_5Template):

    def _remove_history_thinking(self, inputs) -> None:
        return


register_template(
    QwenTemplateMeta(
        'qwen3_5_keep_history_think',
        template_cls=Qwen3_5KeepHistoryThinkTemplate,
        default_system=None,
        thinking_prefix='<think>\n',
        non_thinking_prefix='<think>\n\n</think>\n\n',
        agent_template='qwen3_5',
        is_thinking=True))
