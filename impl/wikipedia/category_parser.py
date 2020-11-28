import util
from collections import defaultdict
import wikitextparser as wtp
from wikitextparser import WikiText, Template
import re
from typing import Optional
from tqdm import tqdm
import multiprocessing as mp


DBPEDIA_PREFIX = 'http://dbpedia.org/resource/'
CATEGORY_PREFIX = 'Category:'
TEMPLATE_PREFIX = 'Template:'
DBPEDIA_TEMPLATE_PREFIX = DBPEDIA_PREFIX + TEMPLATE_PREFIX


def _extract_parent_categories_from_markup(categories_and_templates_markup: tuple) -> dict:
    util.get_logger().info('WIKIPEDIA/CATEGORIES: Extracting parent categories from category markup..')
    categories_markup, templates_markup = categories_and_templates_markup
    template_definitions = _prepare_template_definitions(templates_markup)

    with mp.Pool(processes=util.get_config('max_cpus')) as pool:
        data = [(cat, markup, template_definitions) for cat, markup in categories_markup.items()]
        parent_categories = {cat: parents for cat, parents in tqdm(pool.imap_unordered(_extract_parents_for_category, data, chunksize=1000), total=len(data))}
    return parent_categories


def _extract_parents_for_category(data: tuple) -> tuple:
    cat, markup, template_definitions = data
    content = _replace_templates_in_category(wtp.parse(markup), template_definitions)
    category_targets = {link.target for link in content.wikilinks if link.target.startswith('Category:')}
    category_parents = {DBPEDIA_PREFIX + t.replace(' ', '_') for t in category_targets}
    return cat, category_parents


# PREPARE & APPLY TEMPLATES


def _prepare_template_definitions(templates_markup: dict) -> dict:
    template_definitions = defaultdict(str)
    # extract parts of the template that will be placed on the page the template is applied to
    for name, content in templates_markup.items():
        name = name[len(DBPEDIA_TEMPLATE_PREFIX):].replace('_', ' ')
        name = name[0].upper() + name[1:]
        content = re.sub(r'</?includeonly>', '', content)  # remove includeonly-tags
        content = re.sub(r'<noinclude>(.|\n)*?</noinclude>', '', content)  # remove content in noinclude-tags
        content = _filter_for_onlyinclude(content)
        template_definitions[name] = content
    # handle redirects in templates
    for name in set(template_definitions):
        content = template_definitions[name]
        if content.startswith('#REDIRECT'):
            parse = wtp.parse(content[len('#REDIRECT '):])
            if not parse.wikilinks or not parse.wikilinks[0].target.startswith(TEMPLATE_PREFIX):
                template_definitions[name] = ''
            else:
                target = parse.wikilinks[0].target[len(TEMPLATE_PREFIX):]
                template_definitions[name] = template_definitions[target]
    return template_definitions


ONLYINCLUDE_START_TAG = '<onlyinclude>'
ONLYINCLUDE_END_TAG = '</onlyinclude>'
def _filter_for_onlyinclude(text: str) -> str:
    if ONLYINCLUDE_START_TAG not in text:
        return text
    if ONLYINCLUDE_END_TAG not in text:
        return ''  # start tag without end tag
    onlyinclude_start = text.index(ONLYINCLUDE_START_TAG) + len(ONLYINCLUDE_START_TAG)
    onlyinclude_end = text.index(ONLYINCLUDE_END_TAG)
    return text[onlyinclude_start:onlyinclude_end]


def _replace_templates_in_category(category_content: WikiText, template_definitions: dict) -> WikiText:
    for template in category_content.templates:
        template_content = _get_template_content(template, template_definitions)
        if template_content is None:
            continue
        category_content[slice(*template.span)] = template_content
    return category_content


def _get_template_content(template: Template, template_definitions: dict) -> Optional[str]:
    if not template or not template.string.startswith('{{'):
        return None
    try:
        name = template.normal_name(capitalize=True)
    except IndexError:
        return None
    content = wtp.parse(template_definitions[name])
    content = _apply_parameters(content, _get_template_arguments(template))
    for it in content.templates:
        if not it or not it.string.startswith('{{'):
            continue
        it_name = it.normal_name(capitalize=True)
        it_content = _get_template_content(it, template_definitions) if it_name != name else ''
        content[slice(*it.span)] = it_content
    return content.string


def _get_template_arguments(template: Template) -> dict:
    args = {}
    for arg in template.arguments:
        args[arg.name.strip('\r\n\t ')] = arg.value
    return args


def _apply_parameters(content: WikiText, arguments: dict) -> WikiText:
    for p in content.parameters:
        if not p:
            continue
        param_value = arguments[p.name] if p.name in arguments else _apply_parameters(wtp.parse(p.default or ''), arguments).string
        content[slice(*p.span)] = param_value
    return content
