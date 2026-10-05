"""FB5: actual release UI, numeric limits, draft tabs, responsive bilingual forms."""

from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from playwright.sync_api import sync_playwright,expect
from block_test_artifacts import artifact_path
from block_test_packages import install_test_package
from ui_smoke_common import isolated_server,graph_payload,create_project_api,project_editor_url
from blocs.qwen_tts.block import QwenTtsBlock


def geometry(modal):
    measured=modal.evaluate("""el=>{
      const r=el.getBoundingClientRect(),b=el.querySelector('.owned-body');
      return {inside:r.left>=-1&&r.right<=innerWidth+1&&r.top>=-1&&r.bottom<=innerHeight+1,
        overflow:el.scrollWidth>el.clientWidth+2||b.scrollWidth>b.clientWidth+2,
        controls:[...el.querySelectorAll('[data-close-block-modal],[data-owned-apply]')].every(a=>{
          const q=a.getBoundingClientRect();return q.top>=0&&q.bottom<=innerHeight+1}),
        background:getComputedStyle(el).backgroundColor,
        noErrorVisible:[...el.querySelectorAll('[data-block-modal-no-error]')].every(a=>{
          const q=a.getBoundingClientRect();return q.top>=0&&q.bottom<=innerHeight+1&&q.bottom<=r.bottom+1}),
        unlabeled:[...el.querySelectorAll('input,select,textarea')].filter(c=>!c.labels?.length&&!c.getAttribute('aria-label')).length};
    }""")
    assert measured['inside'] and measured['controls'] and not measured['overflow'] and not measured['unlabeled'],measured
    assert measured['background'] not in {'transparent','rgba(0, 0, 0, 0)'}
    assert measured['noErrorVisible'],measured


def main():
    for origin in ('managed','linked'):
        with isolated_server() as server,sync_playwright() as playwright:
            model=install_test_package(server,'qwen_tts',origin=origin)
            node=QwenTtsBlock().build_node_payload(node_id='ui-case',position={'x':160,'y':160});node['block_version']=model['version']
            project=create_project_api(server,document=graph_payload('Qwen3 TTS settings',[node],[]))['project']
            url=project_editor_url(server.base_url,project['project_id'],workspace_project_id=project['workspace_project_id'])
            browser=playwright.chromium.launch(headless=True)
            try:
                page=browser.new_page(viewport={'width':1440,'height':900})
                page.add_init_script("window.localStorage.setItem('bloxsmith.inspectorPinned','true')")
                errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                page.goto(url);card=page.locator('.canvas-node[data-node-id="ui-case"]')
                card.locator('h3').dblclick();modal=page.locator('.owned-modal')
                field=lambda key:modal.locator('[data-block-config-field="'+key+'"]')
                def save():
                    with page.expect_response(lambda r:r.url.endswith('/ui-action') and r.request.method=='POST') as response:
                        modal.locator('[data-owned-apply]').click()
                    assert not response.value.json().get('error'),response.value.json()
                    if modal.is_visible():
                        modal.locator('[data-close-block-modal]').first.click()
                    page.reload();card.locator('h3').dblclick()
                expect(modal.locator('[data-owned-apply]')).to_be_disabled()
                field('instruct').fill('Speak calmly');modal.locator('[data-close-block-modal]').last.click()
                card.locator('h3').dblclick();expect(field('instruct')).to_have_value('')
                field('instruct').fill('Speak calmly');save()
                expect(field('instruct')).to_have_value('Speak calmly')
                advanced=modal.locator('.owned-advanced').first;advanced.locator('summary').click()
                field('temperature').fill('0.65')
                field('max_pending').fill('0')
                field('threads').fill('0');advanced.locator('summary').click()
                modal.locator('[data-owned-apply]').click();expect(advanced).to_have_attribute('open','')
                expect(field('threads')).to_be_focused();field('threads').fill('2');save()
                advanced.locator('summary').click()
                expect(field('temperature')).to_have_value('0.65')
                expect(field('max_pending')).to_have_value('0')
                field('max_pending').fill('1');save()
                for width,height in ((1440,900),(800,700),(390,740),(320,568)):
                    page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(100)
                    modal.locator('.owned-body').evaluate('el=>{el.scrollTop=0}')
                    geometry(modal);page.screenshot(path=artifact_path('qwen_tts-'+origin+'-'+str(width)+'.png'))
                page.set_viewport_size({'width':1440,'height':900})
                summary=advanced.locator('summary');summary.scroll_into_view_if_needed();summary.focus();summary.press('Enter')
                expect(advanced).to_have_attribute('open','');geometry(modal)
                page.screenshot(path=artifact_path('qwen_tts-'+origin+'-advanced.png'))
                specs=json.loads((Path(__file__).parents[1]/'fields.json').read_text())
                assert {s['key'] for s in specs}==set(model['config'])
                assert modal.locator('[data-block-config-field]').count()==len(specs)
                modal.locator('[data-close-block-modal]').first.click();card.click(position={'x':20,'y':20})
                inspector=page.locator('[data-properties-surface="inspector"][data-node-id="ui-case"]:visible')
                expect(inspector).to_be_visible();assert inspector.evaluate('el=>el.scrollWidth<=el.clientWidth+2')
                panel=inspector.locator('[data-block-config-field="instruct"]');panel.fill('Speak brightly')
                tabs=inspector.locator('[data-owned-inspector-tab="general"]');tabs.focus();tabs.press('ArrowRight')
                expect(inspector.locator('[data-inspector-panel-tab="ports"]')).to_be_visible()
                inspector.locator('[data-owned-inspector-tab="ports"]').press('ArrowLeft')
                expect(panel).to_have_value('Speak brightly')
                with page.expect_response(lambda r:r.url.endswith('/ui-action') and r.request.method=='POST') as response:
                    inspector.locator('[data-owned-apply]').click()
                assert not response.value.json().get('error'),response.value.json()
                page.reload();card.click(position={'x':20,'y':20});assert panel.input_value()=='Speak brightly'
                page.screenshot(path=artifact_path('qwen_tts-'+origin+'-inspector.png'))
                page.goto(server.base_url+'/');page.locator('#homeApplicationSettingsButton').click()
                page.locator('#applicationLanguageSelect').select_option('fr')
                page.wait_for_function("window.CWMessages.getLanguage() === 'fr'")
                page.goto(url);card.locator('h3').dblclick()
                expect(advanced.locator('summary')).to_have_text('Réglages avancés')
                expect(modal.locator('[data-i18n="block.qwen_tts.label_asset_directory"]')).to_have_text('Répertoire du modèle')
                expect(field('language').locator('option[value="French"]')).to_have_text('Français')
                for width,height in ((390,740),(320,568)):
                    page.set_viewport_size({'width':width,'height':height})
                    geometry(modal)
                    page.screenshot(path=artifact_path('qwen_tts-'+origin+'-fr-'+str(width)+'.png'))
                assert not errors,errors
                assert page.evaluate('!window.CWBlockUiBlocks?.qwen_tts')
            finally:
                browser.close()
        print('[ok] Qwen3 TTS '+origin+' responsive bilingual UI and draft tabs',flush=True)


if __name__=='__main__':
    main()
